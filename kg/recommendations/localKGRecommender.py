from .baseKGRecommender import BaseKGRecommender

class LocalKGRecommender(BaseKGRecommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_kg = self.ontology + self.recommendations

    def _execute_query(self, map_kg, map_id):
        combined_kg = self.base_kg + map_kg
        action_mask = [1.0] * self.action_size

        sparql = self.query.replace("MAP_ID_REPLACE", map_id)
        results = combined_kg.query(sparql)

        for row in results:
            idx = row.maskedActionIndex.toPython()
            weight = float(row.maskWeight)
            action_mask[idx] = action_mask[idx] * (1 - weight)

        return action_mask
