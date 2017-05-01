package com.arda.submodular;

/**
 *
 * @author arda
 */
public class Edge<U,I> {
    public U user;
    public I item;
    public Double value;
    public Double trueValue;
    
    public Edge(U user, I item, Double value) {
        this.user = user;
        this.item = item;
        this.value = value;
        this.trueValue = value;
    }
    
    public String toString() {
        return user + " " + item + " " + value;
    }
    
    
}
