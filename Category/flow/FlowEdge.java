package com.arda.flow;

import com.arda.flow.FlowNode;

/**
 *
 * @author arda
 */
public class FlowEdge {
  public int id;
  public FlowNode start;
  public FlowNode end;
  public int cap;
  public double cost;

  public FlowEdge(FlowNode start, FlowNode end, int cap, double cost) {
    this.start = start;
    this.end   = end;
    this.cap   = cap;
    this.cost  = cost;
  }

  public String toString() {
      return String.format("a %1$d %2$d 0 %3$d %4$.4f",start.id, end.id, cap, cost);
  }

}
