from pyoxigraph import Store, NamedNode, Literal as OxLiteral, RdfFormat, Quad
import rdflib
import uuid
from kg.recommendations.baseKGRecommender import BaseKGRecommender

class OxigraphKGRecommender(BaseKGRecommender):
    def __init__(self, ontology_file, recommendation_file, query, domain_uri, env_problem_type, action_size=7):
        self.ontology_file = ontology_file
        self.recommendation_file = recommendation_file
        self.query = query
        self.domain_uri = domain_uri
        self.env_problem_type = env_problem_type
        self.action_size = action_size

        self.domain = rdflib.Namespace(domain_uri)
        self.domain_ns = domain_uri
        self.problem_type = rdflib.URIRef(f"{domain_uri}problemType_{env_problem_type}")
        self.problem_type_str = f"{domain_uri}problemType_{env_problem_type}"
        
        self.base_store = Store()
        self._load_into_store(self.base_store, ontology_file)
        self._load_into_store(self.base_store, recommendation_file)
    
    @staticmethod
    def _load_into_store(store: Store, path: str):
        with open(path, 'r') as f:
            store.load(f.read(), format=RdfFormat.TURTLE)
    
    def _build_map_graph(self, map_array):
        kg_store = Store()
        map_id = f"Map_{uuid.uuid4()}"
        map_uri = f"{self.domain_ns}{map_id}"
        
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

        # Add map triples
        kg_store.add(Quad(
            NamedNode(map_uri),
            NamedNode(rdf_type),
            NamedNode(f"{self.domain_ns}Map")
        ))
        kg_store.add(Quad(
            NamedNode(map_uri),
            NamedNode(f"{self.domain_ns}mapId"),
            OxLiteral(map_id)
        ))
        kg_store.add(Quad(
            NamedNode(map_uri),
            NamedNode(f"{self.domain_ns}hasProblemType"),
            NamedNode(self.problem_type_str)
        ))

        # Add cell triples
        for x, row in enumerate(map_array):
            for y, cell in enumerate(row):
                cell_uri = f"{self.domain_ns}Cell_{x}_{y}_{uuid.uuid4()}"
                
                kg_store.add(Quad(
                    NamedNode(map_uri),
                    NamedNode(f"{self.domain_ns}hasCell"),
                    NamedNode(cell_uri)
                ))
                kg_store.add(Quad(
                    NamedNode(cell_uri),
                    NamedNode(rdf_type),
                    NamedNode(f"{self.domain_ns}Cell")
                ))
                kg_store.add(Quad(
                    NamedNode(cell_uri),
                    NamedNode(f"{self.domain_ns}coordinateX"),
                    OxLiteral(str(x), datatype=NamedNode("http://www.w3.org/2001/XMLSchema#integer"))
                ))
                kg_store.add(Quad(
                    NamedNode(cell_uri),
                    NamedNode(f"{self.domain_ns}coordinateY"),
                    OxLiteral(str(y), datatype=NamedNode("http://www.w3.org/2001/XMLSchema#integer"))
                ))
                kg_store.add(Quad(
                    NamedNode(cell_uri),
                    NamedNode(f"{self.domain_ns}channelValue"),
                    OxLiteral(",".join(map(str, cell)))
                ))

        return kg_store, map_id
    
    def _execute_query(self, map_store, map_id):
        combined_store = Store()
        
        for quad in self.base_store:
            combined_store.add(quad)
        
        for quad in map_store:
            combined_store.add(quad)
        
        action_mask = [1.0] * self.action_size

        sparql = self.query.replace("MAP_ID_REPLACE", map_id)
        
        results = combined_store.query(sparql)
        
        for result in results:
            idx = int(result['maskedActionIndex'].value)
            weight = float(result['maskWeight'].value)
            action_mask[idx] = action_mask[idx] * (1 - weight)
        
        return action_mask
