package com.arda.submodular;

import es.uam.eps.ir.ranksys.examples.Metrics.TUDiv;
import es.uam.eps.ir.ranksys.examples.Metrics.TIDiv;
import edu.stanford.nlp.util.BinaryHeapPriorityQueue;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.rec.Recommender;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author arda
 */
public class SubmodularRecommender<U,I,F1,F2> implements Recommender<U,I> {
    public HashMap<U, Set<F2>> userFeatures = new HashMap();
    public HashMap<I, Set<F1>> itemFeatures = new HashMap();
    public HashMap<U, HashMap<I, Edge>> candidateUserEdges = new HashMap();
    public HashMap<I, HashMap<U, Edge>> candidateItemEdges = new HashMap();
    public HashMap<U, ArrayList<Edge> > solution = new HashMap();
    public List<F2> userFeatureSet = new ArrayList();
    public List<F1> itemFeatureSet = new ArrayList();
    public double lamd = 1;
    public double mu   = .5;
    public int cutoff;

    public SimplePreferenceData<U,I> trainData;
    public SimplePreferenceData<U,I> candidates;
    public SimpleFeatureData<I,F1,Double> itemFeatureData;
    public SimpleFeatureData<U,F2,Double> userFeatureData;
    public TUDiv<U,I,F1> tudiv;
    public TIDiv<U,I,F2> tidiv;
    public HashMap<U,HashMap<F1,Double>> catThreshold;
    public HashMap<I,HashMap<F2,Double>> typeThreshold;
    
    public SubmodularRecommender(SimplePreferenceData<U,I> trainData, SimpleFeatureData<I,F1,Double> itemFeatureData, SimpleFeatureData<U,F2,Double> userFeatureData, SimplePreferenceData<U,I> candidates, int cutoff, double lamd, double mu) throws IOException {
        this.cutoff = cutoff;
        this.trainData = trainData;
        this.userFeatureData = userFeatureData;
        this.itemFeatureData = itemFeatureData;
        this.candidates = candidates;
        this.tudiv = new TUDiv<U,I,F1>(trainData, itemFeatureData, new NoRelevanceModel(), cutoff);
        this.tidiv = new TIDiv<U,I,F2>(trainData, userFeatureData, new NoRelevanceModel(), cutoff);
        this.lamd = lamd;
        this.mu   = mu;
        
        initializeItemFeatures(itemFeatureData);
        initializeUserFeatures(userFeatureData);
        
        itemFeatureSet = itemFeatureData.getAllFeatures().collect(Collectors.toList());
        userFeatureSet = userFeatureData.getAllFeatures().collect(Collectors.toList());
        
        initializeUserEdges(candidates);
        initializeItemEdges(candidates);

        initializeItemThresholds();
        initializeUserThresholds();
    }         

    public void initializeUserThresholds () {
        catThreshold = tudiv.categoryThresholds();
        for (U user: candidateUserEdges.keySet()) {
            for (I item: candidateUserEdges.get(user).keySet()) {
                for (F1 category: itemFeatures.get(item)) {
                    if (catThreshold.get(user).getOrDefault(category,0.0) != 0.0)
                        candidateUserEdges.get(user).get(item).value = ((Double) candidateUserEdges.get(user).get(item).value) + lamd;
                }
            }
        }
    }
    
    public void initializeItemThresholds () {
        typeThreshold = tidiv.typeThresholds();
        for (I item: candidateItemEdges.keySet()) {
            for (U user: candidateItemEdges.get(item).keySet()) {
                for (F2 type: userFeatures.get(user)) {
                    if (typeThreshold.get(item).getOrDefault(type,0.0) != 0.0)
                        candidateUserEdges.get(user).get(item).value = ((Double) candidateUserEdges.get(user).get(item).value) + mu;
                }
            }
        }
    }
    
    public void initializeUserFeatures(SimpleFeatureData<U,F2,Double> userFeatureData) {
        Set<U> users = userFeatureData.itemMap.keySet();
        for (U user: users) {
            userFeatures.put(user, new HashSet());
            List<Tuple2<F2, Double>> uFeatures = userFeatureData.itemMap.getOrDefault(user, new ArrayList<>());
            for (Tuple2<F2,Double> tup: uFeatures)  {
                userFeatures.get(user).add(tup.v1);
            }
        }
    }
    
    public void initializeItemFeatures(SimpleFeatureData<I,F1,Double> itemFeatureData) {
        Set<I> items = itemFeatureData.itemMap.keySet();
        for (I item: items) {
            itemFeatures.put(item, new HashSet());
            List<Tuple2<F1, Double>> uFeatures = itemFeatureData.itemMap.getOrDefault(item, new ArrayList<>());
            for (Tuple2<F1,Double> tup: uFeatures)  {
                itemFeatures.get(item).add(tup.v1);
            }
        }
    }
    
    public void initializeUserEdges(SimplePreferenceData<U, I> candidates) {
        for (U user: candidates.userMap.keySet()) {
            HashMap<I, Edge> userEdges = new HashMap();
            candidateUserEdges.put(user, userEdges);
            solution.put(user, new ArrayList());
            List<IdPref<I>> candidateItems = candidates.userMap.get(user);
            for (IdPref<I> x: candidateItems) {
                I item = x.v1;
                Double rel = x.v2;
                Edge e = new Edge(user, item, rel);
                userEdges.put(item, e);
            }
        }
    }
    public void initializeItemEdges(SimplePreferenceData<U, I> candidates) {
        for (I item: candidates.itemMap.keySet()) {
            HashMap<U, Edge> itemEdges = new HashMap();
            candidateItemEdges.put(item, itemEdges);
            List<IdPref<U>> candidateUsers = candidates.itemMap.get(item);
            for (IdPref<U> x: candidateUsers) {
                U user = x.v1;
                Edge e = candidateUserEdges.get(user).get(item);
                itemEdges.put(user, e);
            }
        }
    }
    
    
    
    public void initialize() throws IOException {
        int c = cutoff;
        HashMap<U, Integer> displayConstraint = new HashMap();
        BinaryHeapPriorityQueue<Edge> heap = new BinaryHeapPriorityQueue();

        for (U user: candidateUserEdges.keySet()) {
            displayConstraint.put(user,c);
            for (Edge e: candidateUserEdges.get(user).values()) {
                heap.add(e, (double)e.value);
            }
        }
        
        int counter = 0;
        while (!heap.isEmpty()) {
            Edge candidate = heap.removeFirst();
            U user = (U) candidate.user;
            I item = (I) candidate.item;
            if (displayConstraint.get(user) > 0) {
                counter ++;
                
                solution.get(user).add(candidate);
                displayConstraint.put(user,displayConstraint.get(user)-1);
                
                // reduce score of items from same category if threshold is met
                Set<F1> categories = itemFeatures.getOrDefault(item, new HashSet());
                for (F1 category: categories) {
                    HashMap<F1, Double> thr = catThreshold.getOrDefault(user, new HashMap());
                    if (thr.getOrDefault(category, 0.0) == 1.0) {
                        //System.out.println("Cat threshold met!");
                        for (Edge e: candidateUserEdges.get(user).values()) {
                            I otherItem = (I) e.item;
                            if (itemFeatures.get(otherItem).contains(category) && heap.contains(e)) {
                                e.value = ((Double) e.value)-lamd;
                                heap.changePriority(e, (double) e.value);
                            }
                        }
                    }
                    if (thr.containsKey(category))
                            thr.put(category, thr.get(category)-1);
                }
                
                // reduce score of users from same type if threshold is met
                Set<F2> types = userFeatures.getOrDefault(user, new HashSet());
                for (F2 type: types) {
                    HashMap<F2, Double> thr = typeThreshold.getOrDefault(item, new HashMap());
                    if (thr.getOrDefault(type, 0.0) == 1.0) {
                        //System.out.println("Type threshold met!");
                        for (Edge e: candidateItemEdges.get(item).values()) {
                            U otherUser = (U) e.user;
                            if (userFeatures.get(otherUser).contains(type) && heap.contains(e)) {
                                e.value = ((Double) e.value)-mu;
                                heap.changePriority(e, (double) e.value);
                            }
                        } 
                    }
                    if (thr.containsKey(type))
                        thr.put(type, thr.get(type)-1);
                }
            }
        }

    }
    
    public void printSolution(String fn) throws IOException {
        File file = new File (fn);
        PrintWriter printWriter = new PrintWriter (fn);
        for (U user: solution.keySet()) {
            List<Edge> userItems = solution.get(user);
            userItems.sort(new Comparator<Edge>() {
                @Override
                public int compare(Edge o1, Edge o2) {
                    return Double.compare(o2.trueValue, o1.trueValue);
                }
            });
            for (Edge entry: userItems) {
                printWriter.print(user + "\t" + entry.item + "\t" + entry.trueValue + "\n");
            }
        }
        printWriter.flush();
        printWriter.close();
    }
    
    public void dumpMap(Map x) {
        for (Object y: x.keySet())
            System.out.print(y+"->"+x.get(y)+",");
        System.out.println();
    }

    @Override
    public Recommendation<U, I> getRecommendation(U u) {
        List<Tuple2od<I>> recList = new ArrayList();
        for (Edge entry: solution.get(u))
            recList.add(new Tuple2od(entry.item, (double) entry.trueValue));
        Collections.sort( recList, new Comparator<Tuple2od>() {
            public int compare( Tuple2od o1, Tuple2od o2 ) {
                return Double.compare(o2.v2, o1.v2);
            }
        } );
        return new Recommendation(u, recList);
    }

    @Override
    public Recommendation<U, I> getRecommendation(U u, int maxLength) {
        return getRecommendation(u);
    }

    @Override
    public Recommendation<U, I> getRecommendation(U u, Predicate<I> filter) {
        return getRecommendation(u);
    }

    @Override
    public Recommendation<U, I> getRecommendation(U u, int maxLength, Predicate<I> filter) {
        return getRecommendation(u);
    }

    @Override
    public Recommendation<U, I> getRecommendation(U u, Stream<I> candidates) {
        return getRecommendation(u);
    }
}
