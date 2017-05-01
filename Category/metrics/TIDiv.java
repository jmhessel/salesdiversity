package com.arda.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.rel.RelevanceModel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author arda
 */
public class TIDiv<U,I,F2> implements SystemMetric {
    public int cutoff;
    public SimplePreferenceData<U,I> trainData;
    public HashMap<I, HashMap<F2, Double>> typeThreshold;
    public SimpleFeatureData<U,F2,Double> userFeatureData;
    public RelevanceModel rel;
    public HashMap<I, ArrayList<U>> solution;
    
    public TIDiv(SimplePreferenceData<U,I> trainData, SimpleFeatureData<U,F2,Double> userFeatureData, RelevanceModel rel, int cutoff) {
        this.trainData = trainData;
        this.typeThreshold = new HashMap();
        this.userFeatureData = userFeatureData;
        this.solution = new HashMap();
        this.cutoff = cutoff;
        this.rel = rel;
        initialize();
    }

    public void initialize() {
        for (I item: trainData.getAllItems().collect(Collectors.toList())) {
            int itemTotal = 0;
            HashMap<F2, Double> m = new HashMap();
            
            for (IdPref<U> idPref: trainData.itemMap.get(item)) {
                U user = idPref.v1;
                for (Tuple2<F2,Double> tup: userFeatureData.itemMap.get(user)) {
                    F2 type = tup.v1;
                    itemTotal+= tup.v2;
                    m.put(type, ((Double) m.getOrDefault(type,0.0))+tup.v2);
                }
            }
            
            HashMap<F2, Double> k = new HashMap();
            HashMap<F2, Double> t = new HashMap();
            double wholePart = 0;
            if (itemTotal > 0) {
                
                int target = (int) (trainData.numUsers() * cutoff * .2 / trainData.numItems());
                List<Map.Entry<F2, Double>> collect = m.entrySet().stream().collect(Collectors.toList());
                for (Map.Entry<F2, Double> entry: collect) {
                    wholePart += Math.floor(entry.getValue() * target / itemTotal);
                    k.put(entry.getKey(), Math.floor(entry.getValue() * target / itemTotal));
                    t.put(entry.getKey(), (entry.getValue() * target / itemTotal) - Math.floor(entry.getValue() * target / itemTotal));
                }
                List<F2> decreasingFractional = new ArrayList(t.keySet());
                Collections.sort(decreasingFractional, new Comparator<F2>(){
                    @Override
                    public int compare(F2 o1, F2 o2) {
                        return Double.compare(t.get(o2),t.get(o1));
                    }
                });
                decreasingFractional.stream().limit((int)(target-wholePart)).forEach(type -> k.put(type,k.get(type)+1.0));
            }
            
            typeThreshold.put(item, k);
        }
    }
    
    public HashMap<I, HashMap<F2, Double>> typeThresholds() {
        return this.typeThreshold;
    }
    
    @Override
    public void add(Recommendation recommendation) {
        U user = (U) recommendation.getUser();
        List<Tuple2od<I>> x = recommendation.getItems();
        x = x.subList(0, x.size() > cutoff ? cutoff : x.size());
        for (Tuple2od<I> y: x) {
            I item = y.v1;
            ArrayList<U> recList = new ArrayList(solution.getOrDefault(item, new ArrayList()));
            if (rel.getModel(user).isRelevant(y.v1))
                recList.add(user);
            solution.put(item, recList);
        }
    }

    @Override
    public double evaluate() {
        HashMap<I, HashMap<F2, Double>> realizedValues = new HashMap();
        for (I item: solution.keySet()) {
            HashMap<F2, Double> k = new HashMap();
            for (U user: solution.get(item)) {
                for (Tuple2<F2,Double> tup: userFeatureData.itemMap.get(user)) {
                    F2 type = tup.v1;
                    k.put(type, k.getOrDefault(type,0.0)+1.0);
                }
            }
            realizedValues.put(item, k);
        }
        
        double num = 0.0;
        double den = 0.0;
        for (I item: typeThreshold.keySet()) {
            for (F2 type: typeThreshold.get(item).keySet()) {
                double realized = (double) realizedValues.getOrDefault(item, new HashMap()).getOrDefault(type, 0.0);
                double desired  = typeThreshold .get(item).get(type);
                if (desired > 0.0) {
                    num += Math.min(realized, desired);
                    den += desired;
                }
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
