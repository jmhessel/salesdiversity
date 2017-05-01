package com.arda.submodular;

import java.util.Comparator;

/**
 *
 * @author arda
 */
public class EdgeComparator implements Comparator<Edge>  {
    @Override
    public int compare(Edge e1, Edge e2) {
        Double d1 = (Double) e1.value;
        Double d2 = (Double) e2.value;
        
        if (d1 > d2)
            return 1;
        else if (d2 > d1)
            return -1;
        return 0;
    }
}
