package com.arda.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.RelevanceModel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author arda
 */
public class TUDiv<U,I,F1> implements SystemMetric {
    public int target;
    public int cutoff;
    public SimplePreferenceData<U,I> trainData;
    public HashMap<U, HashMap<F1, Double>> catThreshold;
    public SimpleFeatureData<I,F1,Double> itemFeatureData;
    public RelevanceModel relTest;
    public HashMap<U, HashSet<I>> solution;
    
    public TUDiv(SimplePreferenceData<U,I> trainData, SimpleFeatureData<I,F1,Double> itemFeatureData, RelevanceModel relTest, int cutoff) {
        this.trainData = trainData;
        this.catThreshold = new HashMap();
        this.itemFeatureData = itemFeatureData;
        this.solution = new HashMap();
        this.cutoff = cutoff;

        this.relTest  = relTest;
        initialize();
    }

    public void initialize() {
        for (U user: trainData.getAllUsers().collect(Collectors.toList())) {
            int userTotal = 0;
            HashMap<F1, Double> m = new HashMap();
            catThreshold.put(user, m);
            for (IdPref<I> idPref: trainData.userMap.get(user)) {
                I item = idPref.v1;
                for (Tuple2<F1,Double> tup: itemFeatureData.itemMap.get(item)) {
                    F1 category = tup.v1;
                    userTotal += tup.v2;
                    m.put(category, ((Double) m.getOrDefault(category,0.0))+tup.v2);
                }
            }
            
            HashMap<F1, Double> k = new HashMap();
            HashMap<F1, Double> t = new HashMap();
            double wholePart = 0;

            List<Map.Entry<F1, Double>> collect = m.entrySet().stream().collect(Collectors.toList());
            for (Map.Entry<F1, Double> entry: collect) {
                wholePart += Math.floor(entry.getValue() * target / userTotal);
                k.put(entry.getKey(), Math.floor(entry.getValue() * target / userTotal));
                t.put(entry.getKey(), (entry.getValue() * target / userTotal) - Math.floor(entry.getValue() * target / userTotal));
            }
            List<F1> decreasingFractional = new ArrayList(t.keySet());
            Collections.sort(decreasingFractional, new Comparator<F1>(){
                @Override
                public int compare(F1 o1, F1 o2) {
                    return Double.compare(t.get(o2),t.get(o1));
                }
            });
            decreasingFractional.stream().limit((int)(target-wholePart)).forEach(type -> k.put(type,k.get(type)+1.0));
            catThreshold.put(user, k);
            
        }
    }
    
    public HashMap<U, HashMap<F1, Double>> categoryThresholds() {
        return this.catThreshold;
    }
    
    @Override
    public void add(Recommendation recommendation) {
        U user = (U) recommendation.getUser();
        HashSet<I> recList = solution.getOrDefault(user, new HashSet());
        List<Tuple2od<I>> x = recommendation.getItems();
        x = x.subList(0, x.size() > cutoff ? cutoff : x.size());
        for (Tuple2od<I> y: x)
            if (relTest.getModel(user).isRelevant(y.v1))
                recList.add(y.v1);
        solution.put(user, recList);
    }

    @Override
    public double evaluate() {
        HashMap<U, HashMap<F1, Double>> realizedValues = new HashMap();
        for (U user: solution.keySet()) {
            HashMap<F1, Double> k = new HashMap();
            for (I item: solution.get(user)) {
                for (Tuple2<F1,Double> tup: itemFeatureData.itemMap.get(item)) {
                    F1 category = tup.v1;
                    k.put(category, k.getOrDefault(category,0.0)+1.0);
                }
            }
            realizedValues.put(user, k);
        }
        
        double num = 0.0;
        double den = 0.0;
        for (U user: catThreshold.keySet()) {
            for (F1 category: catThreshold.get(user).keySet()) {
                double realized = (double) realizedValues.getOrDefault(user, new HashMap()).getOrDefault(category, 0.0);
                double desired  = catThreshold  .get(user).get(category);
                num += Math.min(realized,desired);
                den += desired;
            }
        }
        return num/den;
    }

    @Override
    public void combine(SystemMetric other) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void reset() {
        solution.keySet().stream().forEach(k -> solution.get(k).clear());
    }
     

}