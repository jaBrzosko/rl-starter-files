from kg.recommendations.localKGRecommender import LocalKGRecommender as Local
from kg.recommendations.fusekiKGRecommender import FusekiKGRecommender as Fuseki
from kg.recommendations.fusekiKGOptimizedRecommender import FusekiKGOptimizedRecommender as FusekiOptimized
from kg.recommendations.oxigraphKGRecommender import OxigraphKGRecommender as Oxigraph
import torch
from enum import Enum

class RecommendationMode(Enum):
    IN_MEMORY = 1
    FUSEKI = 2
    FUSEKI_OPTIMIZED = 3
    OXIGRAPH = 4

class KGActionMasker:
    def __init__(self,
                 ontology_file,
                 recommendation_file,
                 query,
                 domain_uri,
                 env_problem_type,
                 action_size=7,
                 recommendation_mode=RecommendationMode.IN_MEMORY,      
                 ):
        
        if recommendation_mode == RecommendationMode.IN_MEMORY:
            self.recommender = Local(
                ontology_file=ontology_file,
                recommendation_file=recommendation_file,
                query=query,
                domain_uri=domain_uri,
                env_problem_type=env_problem_type,
                action_size=action_size,
            )
        elif recommendation_mode == RecommendationMode.FUSEKI:
            self.recommender = Fuseki(
                ontology_file=ontology_file,
                recommendation_file=recommendation_file,
                query=query,
                domain_uri=domain_uri,
                env_problem_type=env_problem_type,
                action_size=action_size,
            )
        elif recommendation_mode == RecommendationMode.FUSEKI_OPTIMIZED:
            self.recommender = FusekiOptimized(
                ontology_file=ontology_file,
                recommendation_file=recommendation_file,
                query=query,
                domain_uri=domain_uri,
                env_problem_type=env_problem_type,
                action_size=action_size,
            )
        elif recommendation_mode == RecommendationMode.OXIGRAPH:
            self.recommender = Oxigraph(
                ontology_file=ontology_file,
                recommendation_file=recommendation_file,
                query=query,
                domain_uri=domain_uri,
                env_problem_type=env_problem_type,
                action_size=action_size,
            )
        else:
            raise ValueError(f"Unknown recommendation mode: {recommendation_mode}")

    def get_action_mask(self, observation, device):
        masks_raw = []
        for obs in observation.image:
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().detach().numpy()
            action_mask = self.recommender.get_action_mask(obs)
            masks_raw.append(action_mask)

        masks_tensor = torch.tensor(masks_raw, device=device, dtype=torch.float)
        return masks_tensor
