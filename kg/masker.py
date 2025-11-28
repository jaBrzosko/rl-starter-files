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

class MaskingOptions:
    def __init__(self,
                probability: float = 1.0,
                episode_cutoff: int = None,
                value_cutoff: float = None,
                ):
        self.probability = probability
        self.episode_cutoff = episode_cutoff
        self.value_cutoff = value_cutoff

class KGActionMasker:
    def __init__(self,
                 ontology_file,
                 recommendation_file,
                 query,
                 domain_uri,
                 env_problem_type,
                 action_size=7,
                 recommendation_mode=RecommendationMode.IN_MEMORY,
                 masking_options: MaskingOptions = MaskingOptions(),
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
        
        self.masking_options = masking_options
        self.episode_number = 0
        self.last_episode_value = None
        self.best_episode_value = None

    def get_action_mask(self, observation, device):
        n = len(observation.image)

        if self.masking_options.episode_cutoff is not None:
            if self.episode_number > self.masking_options.episode_cutoff:
                return torch.ones((n, self.recommender.action_size), device=device, dtype=torch.float)
        if self.masking_options.value_cutoff is not None:
            if self.best_episode_value is not None and self.best_episode_value >= self.masking_options.value_cutoff:
                return torch.ones((n, self.recommender.action_size), device=device, dtype=torch.float)

        masks_raw = []
        for obs in observation.image:
            if self.masking_options.probability < 1.0:
                episode_prob = self.masking_options.probability ** (self.episode_number + 1)
                if torch.rand(1).item() > episode_prob:
                    masks_raw.append([1] * self.recommender.action_size)
                    continue
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().detach().numpy()
            action_mask = self.recommender.get_action_mask(obs)
            masks_raw.append(action_mask)

        masks_tensor = torch.tensor(masks_raw, device=device, dtype=torch.float)
        return masks_tensor

    def set_episode(self, episode_number):
        self.episode_number = episode_number

    def set_episode_value(self, value):
        self.last_episode_value = value
        if self.best_episode_value is None or value > self.best_episode_value:
            self.best_episode_value = value
