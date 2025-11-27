from kg.masker import KGActionMasker, RecommendationMode
import unittest
import torch_ac
import torch

ONTOLOGY_FILE = "kg/data/minigrid_ontology.ttl"
ACTION_SIZE = 7

UNSEEN = [0.0, 0.0, 0.0]
WALL = [2.0, 5.0, 0.0]
EMPTY = [1.0, 0.0, 0.0]
AGENT = [1.0, 0.0, 0.0] # Is the same as EMPTY, but helps to clarify tests
LAVA = [9.0, 0.0, 0.0]
GOAL = [8.0, 1.0, 0.0]

def create_masker(recommendation_file, query, problem_type):
    masker = KGActionMasker(
        ontology_file=ONTOLOGY_FILE,
        recommendation_file=recommendation_file,
        query=query,
        domain_uri="http://example.org/minigrid#",
        env_problem_type=problem_type,
        action_size=ACTION_SIZE,
        recommendation_mode=RecommendationMode.OXIGRAPH,
    )

    return masker

# Utility to rotate map (in-place)
# Necessary because the observations come this weirdly rotated
def rotate_map(map):
    for i in range(len(map)):
        for j in range(i, len(map)):
            map[i][j], map[j][i] = map[j][i], map[i][j]

class TestSimpleLavaKGValidation(unittest.TestCase):
    def setUp(self):
        recommendation_file = "kg/data/experiments/minigrid_recommendations_lava_simple.ttl"
        problem_type = "crossing"

        with open("kg/data/recommendation_query.rq", "r") as f:
            query = f.read()

        self.masker = create_masker(
            recommendation_file=recommendation_file,
            query=query,
            problem_type=problem_type,
        )

    def test_check_unavailable_action_mask(self):
        map = [
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, WALL,   WALL,   WALL,   WALL,   WALL,   WALL],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  AGENT,  EMPTY,  EMPTY, EMPTY],
        ]
        rotate_map(map)

        obs = torch_ac.DictList({
            "image": [map],
        })

        action_mask = self.masker.get_action_mask(obs, device=torch.device("cpu"))

        expected_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

        self.assertTrue(torch.equal(action_mask, expected_mask), f"Expected mask: {expected_mask}, but got: {action_mask}")

    def test_check_lava_ahead(self):
        map = [
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, WALL,   WALL,   WALL,   WALL,   WALL,   WALL],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  LAVA,   EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  AGENT,  EMPTY,  EMPTY, EMPTY],
        ]
        rotate_map(map)

        obs = torch_ac.DictList({
            "image": [map],
        })

        action_mask = self.masker.get_action_mask(obs, device=torch.device("cpu"))

        expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

        self.assertTrue(torch.equal(action_mask, expected_mask), f"Expected mask: {expected_mask}, but got: {action_mask}")

    def test_check_wall_ahead(self):
        map = [
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, WALL,   WALL,   WALL,   WALL,   WALL,   WALL],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  WALL,   EMPTY,  EMPTY, EMPTY],
            [UNSEEN, WALL,   EMPTY,  AGENT,  EMPTY,  EMPTY, EMPTY],
        ]
        rotate_map(map)

        obs = torch_ac.DictList({
            "image": [map],
        })

        action_mask = self.masker.get_action_mask(obs, device=torch.device("cpu"))

        expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

        self.assertTrue(torch.equal(action_mask, expected_mask), f"Expected mask: {expected_mask}, but got: {action_mask}")

    def test_check_empty_ahead(self):
        map = [
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN],
            [UNSEEN, WALL,   WALL,   WALL,   WALL,   WALL,   WALL],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY,  EMPTY],
            [UNSEEN, WALL,   EMPTY,  EMPTY,  EMPTY,  EMPTY,  EMPTY],
            [UNSEEN, WALL,   WALL,   EMPTY,  LAVA,   LAVA,   LAVA],
            [UNSEEN, WALL,   EMPTY,  AGENT,  EMPTY,  EMPTY,  EMPTY],
        ]
        rotate_map(map)

        obs = torch_ac.DictList({
            "image": [map],
        })

        action_mask = self.masker.get_action_mask(obs, device=torch.device("cpu"))

        expected_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

        self.assertTrue(torch.equal(action_mask, expected_mask), f"Expected mask: {expected_mask}, but got: {action_mask}")

if __name__ == "__main__":
    unittest.main()
