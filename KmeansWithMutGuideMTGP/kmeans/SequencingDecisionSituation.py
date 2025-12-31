import KmeansWithMutGuideMTGP.kmeans.DecisionSituation as DecisionSituation

class SequencingDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, data, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self, data)

    def clone(self):
        dataClone = []
        for op in self.data:
            dataClone.append(op)
        return SequencingDecisionSituation(dataClone)


