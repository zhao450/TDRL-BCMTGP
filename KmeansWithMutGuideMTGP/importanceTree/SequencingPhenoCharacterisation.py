import KmeansWithMutGuideMTGP.importanceTree.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import sequencing


class SequencingPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations
        self.referenceIndexes = []
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.referenceIndexes = []
        for i in range(len(self.decisionSituations)):
            sequencingDecision = self.decisionSituations[i].clone()
            sequencing_data = sequencingDecision.getData()
            ranks = sequencing.GP_evolve_S_ranks(sequencing_data, self.referenceRule)
            self.referenceIndexes.append(ranks)
    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            sequencingDecision = self.decisionSituations[i].clone()
            sequencing_data = sequencingDecision.getData()
            ranks_rule = sequencing.GP_evolve_S_ranks(sequencing_data, rule)
            idxBest = 0
            for j in range(len(ranks_rule)):
                if ranks_rule[j] < ranks_rule[idxBest]:
                    idxBest = j

            if len(self.decisionSituations) < 20 or len(self.referenceIndexes) < 20 or len(self.referenceIndexes[i]) < 3:
                print("len(self.decisionSituations): " + str(len(self.decisionSituations)))
                print("len(self.referenceIndexes): " + str(len(self.referenceIndexes)))
                print("len(self.referenceIndexes[i]): " + str(len(self.referenceIndexes[i])))
            charlist.append(self.referenceIndexes[i][idxBest])

        return charlist
    
    def characterise_returnall(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            sequencingDecision = self.decisionSituations[i].clone()
            sequencing_data = sequencingDecision.getData()
            ranks_rule = sequencing.GP_evolve_S_ranks(sequencing_data, rule)
            charlist.append(ranks_rule)

        return charlist


