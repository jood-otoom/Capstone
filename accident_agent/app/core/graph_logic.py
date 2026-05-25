# app/core/graph_logic.py
import networkx as nx

class TrafficLogicGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_hierarchy()

    def _build_hierarchy(self):
        """
        Builds the 'Yields To' relationships.
        Node A -> Node B means Node A must yield to Node B.
        """
        # 1. Default Road Rules
        self.graph.add_edge("Side Road", "Main Road")
        self.graph.add_edge("Dirt Road", "Paved Road")
        self.graph.add_edge("Entering Roundabout", "Inside Roundabout")
        
        # 2. Traffic Signs (Overrides Default Rules)
        self.graph.add_edge("Yield Sign", "Main Road")
        self.graph.add_edge("Stop Sign", "Main Road")
        self.graph.add_edge("Stop Sign", "Yield Sign") # Stop is lower priority than yield
        
        # 3. Traffic Lights (Overrides Signs)
        self.graph.add_edge("Red Light", "Green Light")
        
        # 4. Officer (Overrides Everything - implemented in logic below)

    def determine_liability(self, vehicle_a_state, vehicle_b_state):
        """
        Traverses the graph to see who owes the right of way.
        """
        if vehicle_a_state == "Officer Signal" or vehicle_b_state == "Officer Signal":
            return "Manual override: Traffic Officer determines right of way."

        # Check if A yields to B
        if nx.has_path(self.graph, vehicle_a_state, vehicle_b_state):
            return "Vehicle A failed to yield right of way."
            
        # Check if B yields to A
        if nx.has_path(self.graph, vehicle_b_state, vehicle_a_state):
            return "Vehicle B failed to yield right of way."
            
        return "Unclear from graph topology. Simultaneous fault or missing visual data."