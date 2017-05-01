package com.arda.flow;

import Flow.FlowEdge;
import Flow.FlowNode;
import es.uam.eps.ir.ranksys.examples.Metrics.TUDiv;
import es.uam.eps.ir.ranksys.examples.Metrics.TIDiv;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.examples.Submodular.Edge;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.rec.Recommender;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
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
public class FlowRecommender<U,I,F1,F2> implements Recommender<U,I> {
    public static int infinity = 10000;
    public HashMap<U, F2> typeMap = new HashMap();
    public HashMap<I, F1> catMap  = new HashMap();
    public HashMap<U, HashMap<I, Edge>> candidateUserEdges = new HashMap();
    public HashMap<I, HashMap<U, Edge>> candidateItemEdges = new HashMap();
    public HashMap<U, List<I>> solution = new HashMap();
    public HashMap<Integer, U> nodeIdToUserMap = new HashMap();
    public HashMap<Integer, I> nodeIdToItemMap = new HashMap();
    public List<F2> userFeatureSet = new ArrayList();
    public List<F1> itemFeatureSet = new ArrayList();
    public List<U>  targetUsers    = new ArrayList();
    public double lamd,mu;
    public int cutoff;

    HashMap<I,FlowNode> itemNodes            = new HashMap();
    HashMap<U,FlowNode> userNodes            = new HashMap();
    ArrayList<FlowEdge> allEdges             = new ArrayList();
    ArrayList<FlowNode> allNodes             = new ArrayList();
    
    
    public SimplePreferenceData<U,I> trainData;
    public SimplePreferenceData<U,I> candidates;
    public SimpleFeatureData<I,F1,Double> itemFeatureData;
    public SimpleFeatureData<U,F2,Double> userFeatureData;
    public TUDiv<U,I,F1> tudiv;
    public TIDiv<U,I,F2> tidiv;
    public HashMap<U,HashMap<F1,Double>> catThreshold = new HashMap();
    public HashMap<I,HashMap<F2,Double>> typeThreshold = new HashMap();
    
    public FlowRecommender(SimplePreferenceData<U,I> trainData, SimpleFeatureData<I,F1,Double> itemFeatureData, SimpleFeatureData<U,F2,Double> userFeatureData, SimplePreferenceData<U,I> candidates, int cutoff, double lamd, double mu) throws IOException {
        this.cutoff = cutoff;
        this.trainData = trainData;
        this.userFeatureData = userFeatureData;
        this.itemFeatureData = itemFeatureData;
        this.candidates = candidates;
        this.tudiv = new TUDiv(trainData, itemFeatureData, new NoRelevanceModel(), cutoff);
        this.tidiv = new TIDiv(trainData, userFeatureData, new NoRelevanceModel(), cutoff);
        this.lamd = lamd;
        this.mu   = mu;
        
        initializeItemFeatures(itemFeatureData);
        initializeUserFeatures(userFeatureData);
        
        itemFeatureSet = itemFeatureData.getAllFeatures().collect(Collectors.toList());
        userFeatureSet = userFeatureData.getAllFeatures().collect(Collectors.toList());
        
        initializeUserEdges(candidates);
        initializeItemEdges(candidates);

        targetUsers = new ArrayList(candidateUserEdges.keySet()).subList(0, candidateUserEdges.size());
        
        initializeItemThresholds();
        initializeUserThresholds();
    }    

    public void initializeUserThresholds () {
        catThreshold = tudiv.categoryThresholds();
    }

    public void initializeItemThresholds () {
        typeThreshold = tidiv.typeThresholds();
    }

    public void initializeUserFeatures(SimpleFeatureData<U,F2,Double> userFeatureData) {
        Set<U> users = userFeatureData.itemMap.keySet();
        for (U user: users) {
            List<Tuple2<F2, Double>> uFeatures = userFeatureData.itemMap.getOrDefault(user, new ArrayList<>());
            for (Tuple2<F2,Double> tup: uFeatures)  {
                typeMap.put(user,tup.v1);
            }
        }
    }
    
    public void initializeItemFeatures(SimpleFeatureData<I,F1,Double> itemFeatureData) {
        Set<I> items = itemFeatureData.itemMap.keySet();
        for (I item: items) {
            List<Tuple2<F1, Double>> iFeatures = itemFeatureData.itemMap.getOrDefault(item, new ArrayList<>());
            for (Tuple2<F1,Double> tup: iFeatures)  {
                catMap.put(item,tup.v1);
                break;
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
    
    
    
    public void buildNetwork() throws IOException {
        HashMap<U,HashMap<F1, FlowNode>> nPrime  = new HashMap();
        HashMap<U,HashMap<F1, FlowNode>> n       = new HashMap();
        HashMap<I,HashMap<F2, FlowNode>> mPrime  = new HashMap();
        HashMap<I,HashMap<F2, FlowNode>> m       = new HashMap();
        FlowNode sink = new FlowNode(-targetUsers.size()*cutoff);
        allNodes.add(sink);
        
        
        
        for (U user: targetUsers) {
            FlowNode userNode = new FlowNode(cutoff);
            allNodes.add(userNode);
            userNodes.put(user, userNode);
            nodeIdToUserMap.put(userNode.id, user);
            HashMap<F1, FlowNode> x = new HashMap();
            HashMap<F1, FlowNode> y = new HashMap();
            for (F1 category: catThreshold.get(user).keySet()) {
                if (catThreshold.get(user).get(category) != 0.0) {
                    FlowNode nPrimeNode = new FlowNode();
                    FlowNode nNode      = new FlowNode();
                    nodeIdToUserMap.put(nNode.id, user);

                    FlowEdge e1 = new FlowEdge(userNode  , nPrimeNode, (int)Math.round(catThreshold.get(user).get(category)), 0);
                    FlowEdge e2 = new FlowEdge(userNode  , nNode     , infinity                                             , 0);
                    FlowEdge e3 = new FlowEdge(nPrimeNode, nNode     , (int)Math.round(catThreshold.get(user).get(category)), -lamd);

                    x.put(category, nNode);
                    y.put(category, nPrimeNode);

                    allEdges.add(e1);
                    allEdges.add(e2);
                    allEdges.add(e3);
                    
                    allNodes.add(nNode);
                    allNodes.add(nPrimeNode);
                }
            }
            nPrime.put(user, y);
            n     .put(user, x);
        }
        
        for (I item: candidateItemEdges.keySet()) {
            FlowNode itemNode = new FlowNode();
            allNodes.add(itemNode);
            itemNodes.put(item, itemNode);
            nodeIdToItemMap.put(itemNode.id, item);
            HashMap<F2, FlowNode> x = new HashMap();
            HashMap<F2, FlowNode> y = new HashMap();
            for (F2 type: typeThreshold.get(item).keySet()) {
                if (typeThreshold.get(item).get(type) != 0.0) {
                    FlowNode mPrimeNode = new FlowNode();
                    FlowNode mNode      = new FlowNode();
                    nodeIdToItemMap.put(mNode.id, item);
                    
                    FlowEdge e1 = new FlowEdge(mPrimeNode, itemNode  , (int)Math.round(typeThreshold.get(item).get(type)), 0);
                    FlowEdge e2 = new FlowEdge(mNode     , mPrimeNode, (int)Math.round(typeThreshold.get(item).get(type)), -mu);
                    FlowEdge e3 = new FlowEdge(mNode     , itemNode  , infinity                                          , 0);
                    
                    x.put(type, mNode);
                    y.put(type, mPrimeNode);
                    
                    allEdges.add(e1);
                    allEdges.add(e2);
                    allEdges.add(e3);
                    
                    allNodes.add(mNode);
                    allNodes.add(mPrimeNode);
                }
            }
            
            FlowEdge e4 = new FlowEdge(itemNode, sink, infinity, 0);
            allEdges.add(e4);
            mPrime.put(item, y);
            m     .put(item, x);
        }
        
        int totalSpecial = 0;
        for (U user: targetUsers) {
            FlowNode userNode = userNodes.get(user);
            F2 type = typeMap.get(user);
            for (I item: candidateUserEdges.get(user).keySet()) {
                FlowNode itemNode = itemNodes.get(item);
                F1 category = catMap.get(item);
                boolean special1 = false;
                boolean special2 = false;
                if (n.get(user).containsKey(category)) {
                    userNode = n.get(user).get(category);
                    special1 = true;
                }
                if (m.get(item).containsKey(type)) {
                    itemNode = m.get(item).get(type);
                    special2 = true;
                }
                
                if (special1 && special2) totalSpecial ++;
                Double y = -(Double)candidateUserEdges.get(user).get(item).trueValue;
                FlowEdge e1 = new FlowEdge(userNode, itemNode, 1, y );
                allEdges.add(e1);
            }
        }
    }
    
    public void initialize() throws IOException {
        FlowNode.idCounter = 1;
        buildNetwork();
        for (U user: targetUsers) {
            solution.put(user, new ArrayList());
        }
        printProblem("/Users/arda/desktop/datasets/ml-1m2/my_flow.dmx");
        String command = "/Users/arda/desktop/datasets/ml-1m2/MyMCFSolve /Users/arda/desktop/datasets/ml-1m2/my_flow.dmx";
        Process proc = Runtime.getRuntime().exec(command);
        BufferedReader reader =  
              new BufferedReader(new InputStreamReader(proc.getInputStream()));
        
        reader.readLine();
        String line = "";
        while((line = reader.readLine()) != null) {
            try {
                String[] data = line.split(" ");
                int startId = Integer.parseInt(data[0]);
                int endId   = Integer.parseInt(data[1]);
                U user; I item;
                
                if (nodeIdToUserMap.containsKey(startId)) {
                    user = nodeIdToUserMap.get(startId);
                } else {
                    continue;
                }
                
                if (nodeIdToItemMap.containsKey(endId)) {
                    item = nodeIdToItemMap.get(endId);
                } else {
                    continue;
                }
                solution.get(user).add(item);
            } catch (NumberFormatException e) {
                continue;
            } catch (ArrayIndexOutOfBoundsException e) {
                continue;
            }
        }
    }
    
    public void printProblem(String fn) throws IOException {
        BufferedWriter br = new BufferedWriter(new FileWriter(fn));
        br.write(String.format("p min %1$d %2$d",allNodes.size(), allEdges.size()));
        br.newLine();
        for (FlowNode n1: allNodes) {
            if (n1.demand != 0) {
                br.write(n1.toString());
                br.newLine();
            }
        }
        
        for (FlowEdge e1: allEdges) {
            br.write(e1.toString());
            br.newLine();
        }
        br.flush();
    }
    
    public void printSolution(String fn) throws IOException {
        File file = new File (fn);
        PrintWriter printWriter = new PrintWriter (fn);
        for (U user: targetUsers) {
            for (I item: solution.get(user)) {          
                printWriter.print(user + "\t" + item + "\t" + candidateUserEdges.get(user).get(item).value + "\n");
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
        for (I item: solution.get(u))
            recList.add(new Tuple2od(item, (double) candidateUserEdges.get(u).get(item).trueValue));
        Collections.sort( recList, new Comparator<Tuple2od>()
        {
            public int compare( Tuple2od o1, Tuple2od o2 )
            {
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
