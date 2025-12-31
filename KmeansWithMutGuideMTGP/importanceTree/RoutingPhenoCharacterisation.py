import KmeansWithMutGuideMTGP.importanceTree.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import routing

class RoutingPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations
        self.referenceIndexes = []
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.referenceIndexes = []
        for i in range(len(self.decisionSituations)):
            routingDecision = self.decisionSituations[i].clone()
            routing_data = routingDecision.getData()
            ranks = routing.GP_evolve_R_ranks(self.referenceRule, routing_data[0], routing_data[1], routing_data[2], routing_data[3], routing_data[4],routing_data[5], routing_data[6], routing_data[7], routing_data[8], routing_data[9])
            self.referenceIndexes.append(ranks)

    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        charlist = []
        for i in range(len(self.decisionSituations)):
            routingDecision = self.decisionSituations[i].clone()
            routing_data = routingDecision.getData()
            ranks_rule = routing.GP_evolve_R_ranks(rule, routing_data[0], routing_data[1], routing_data[2], routing_data[3], routing_data[4],
                                                   routing_data[5], routing_data[6], routing_data[7], routing_data[8], routing_data[9])
            idxBest = 0
            for j in range(len(ranks_rule)):
                if ranks_rule[j] < ranks_rule[idxBest]:
                    idxBest = j
            charlist.append(self.referenceIndexes[i][idxBest])

        return charlist
    def characterise_returnall(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            routingDecision = self.decisionSituations[i].clone()
            routing_data = routingDecision.getData()
            ranks_rule = routing.GP_evolve_R_ranks_change(rule, routing_data[0], routing_data[1], routing_data[2], routing_data[3], routing_data[4],
                                                   routing_data[5], routing_data[6], routing_data[7], routing_data[8], routing_data[9])
            charlist.append(ranks_rule)

        return charlist


