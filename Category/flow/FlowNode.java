package com.arda.flow;

/**
 *
 * @author arda
 */
public class FlowNode {
  public static int idCounter = 1;
  public int id;
  public int demand = 0;
  
  public FlowNode() {
    this.id = idCounter++;
  }

  public FlowNode(int demand) {
    this(); 
    this.demand = demand;
  }

  public String toString() {
      return String.format("n %1$d %2$d", id, demand);
  }
}