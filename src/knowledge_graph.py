import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from src.database import DatabaseManager

class KnowledgeGraphBuilder:
    def __init__(self):
        self.db = DatabaseManager()
        self.graph = nx.Graph()
    
    def build_entity_relationships(self):
        """Build relationships between entities based on co-occurrence"""
        cursor = self.db.conn.cursor()
        
        # Get all entities
        cursor.execute("SELECT id, name, type FROM entities")
        entities = cursor.fetchall()
        
        # Add nodes to graph
        for entity in entities:
            self.graph.add_node(entity['id'], 
                              name=entity['name'], 
                              type=entity['type'])
        
        # Build relationships based on co-occurrence in conversations
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities co-occur in conversations
                cursor.execute("""
                    SELECT COUNT(*) as co_occurrence FROM conversations
                    WHERE (user_input ILIKE %s OR bot_response ILIKE %s)
                    AND (user_input ILIKE %s OR bot_response ILIKE %s)
                """, (f"%{entity1['name']}%", f"%{entity1['name']}%",
                      f"%{entity2['name']}%", f"%{entity2['name']}%"))
                
                result = cursor.fetchone()
                co_occurrence = result['co_occurrence']
                
                if co_occurrence > 0:
                    weight = co_occurrence / 10.0  # Normalize weight
                    self.graph.add_edge(entity1['id'], entity2['id'], weight=weight)
                    
                    # Store relationship in database
                    cursor.execute("""
                        INSERT INTO relationships 
                        (source_entity_id, target_entity_id, relationship_type, weight)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (entity1['id'], entity2['id'], 'co_occurrence', weight))
        
        self.db.conn.commit()
        cursor.close()
    
    def visualize_graph(self, output_path: str = "knowledge_graph.png"):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(15, 10))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Draw nodes by type
        node_colors = {
            'PRODUCT': 'lightblue',
            'ISSUE': 'lightcoral',
            'ACTION': 'lightgreen',
            'EMOTION': 'lightyellow'
        }
        
        for node_type, color in node_colors.items():
            nodes = [n for n in self.graph.nodes() 
                    if self.graph.nodes[n].get('type') == node_type]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, 
                                 node_color=color, node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=2)
        
        # Draw labels
        labels = {n: self.graph.nodes[n]['name'] for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Customer Support Knowledge Graph", size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()