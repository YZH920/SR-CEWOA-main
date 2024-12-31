import torch

from src.loadData import loadDataOther, loadDataPN
import numpy as np
import json
import time
import math
from scipy.special import gamma


class BWO_:
    def __init__(self, services, constraints, solution=None, popSize=50, MAX_Iter=150, y=""):

        self.pe = 0.2
        self.bestFitnesses = []

        # print(solution)

        if solution is not None:

            for i in range(len(services)):
                for j in range(len(services[i])):
                    services[i][j] = list(services[i][j])
                    services[i][j][0] = round(services[i][j][0], 5)
                    services[i][j][1] = round(services[i][j][1], 5)
                    services[i][j][2] = round(services[i][j][2], 5)
                    services[i][j][3] = round(services[i][j][3], 5)
            for i in range(len(solution)):
                solution[i][0] = round(solution[i][0], 5)
                solution[i][1] = round(solution[i][1], 5)
                solution[i][2] = round(solution[i][2], 5)
                solution[i][3] = round(solution[i][3], 5)
                # normal

                if solution[i] == [0.05314, 0.55528, 0.94008, 0.95495]:
                    solution[i][1] = 0.55527
                if solution[i] == [0.03922, 0.56097, 0.94131, 0.92804]:
                    solution[i][1] = 0.56096
                if solution[i] == [0.17292, 0.5995, 0.92651, 0.92459]:
                    solution[i][2] = 0.92652
                if solution[i] == [0.33474, 0.55123, 0.90018, 0.97161]:
                    solution[i][3] = 0.9716
                if solution[i] == [0.73066, 0.40995, 0.90016, 0.92941]:
                    solution[i][3] = 0.92942

                # qws
                if solution[i] == [0.16904, 0.60902, 0.93639, 0.97272]:
                    solution[i][2] = 0.9364

        self.services = services
        self.constraints = constraints
        self.popSize = popSize
        self.qosNum = 4
        self.MAX_Iter = MAX_Iter
        self.consNum = 2


        # initial
        self.pops = []
        for i in range(self.popSize):
            self.pops.append([np.random.choice(list(range(len(service)))) for service in self.services])
        self.popServices = []

        if solution is not None:
            violate, objFunc, _ = self.calc(solution)
            self.bestFitness = violate + objFunc
            self.bestSolutions = solution
            self.bestPops = []
            for l in range(len(solution)):
                service1 = self.services[l]
                service2 = solution[l]
                try:
                    self.bestPops.append(service1.index(tuple(service2)))
                except:
                    self.services[l].append(tuple(service2))
                    service1 = self.services[l]
                    self.bestPops.append(service1.index(tuple(service2)))
            self.initFitness = self.bestFitness
        else:
            self.bestFitness = 3
            self.bestSolutions = None
            self.bestPops = None
            self.initFitness = 3
        self.initPops = self.bestPops

        for i in range(self.popSize):
            service = [self.services[j][self.pops[i][j]] for j in range(len(self.pops[i]))]
            self.popServices.append(service)
            violate, objFunc, _ = self.calc(service)
            fitness = violate + objFunc
            if self.bestFitness > fitness:
                self.bestFitness = fitness
                self.bestSolutions = service
                self.bestPops = self.pops[i]

    def calc(self, services):

        violate = 0
        serviceNum = 0
        violateConstraints = []
        indicator = [np.array([services[i][j] for i in range(len(services))]) for j in
                     range(self.qosNum)]
        conValues = [np.cumprod(indicator[i + 2])[-1] for i in range(self.consNum)]


        for i in range(len(self.constraints)):
            for constraint in self.constraints[i]:
                if conValues[i] < constraint[-2] or conValues[i] > constraint[-1]:
                    violate += 1
                    violateConstraints.append([i, constraint])
        for i in range(len(services)):
            if services[i][0] > 0:
                serviceNum += 1

        objFunc = (np.sum(indicator[0]) / serviceNum + 1 - np.min(indicator[1])) / 2
        objFunc = float(objFunc)
        return violate, objFunc, violateConstraints

    def start(self):
        t = 0

        g = np.random.random()

        chang=29
        while t < self.MAX_Iter:
            for i in range(self.popSize):

                if self.pops[i] is None :
                    continue
                b0=np.random.random(self.popSize)
                Bf=b0*(1-t/2*self.MAX_Iter)
                wf=0.1 - 0.05 *(t/self.MAX_Iter)
                r1 = np.random.random()
                pop_ = None
                if Bf[i] >0.5:
                    while r1 == 0:
                        r1 = np.random.random()
                    r2 = np.random.random()
                    while r2 == 0:
                        r2 = np.random.random()
                    rand = np.random.randint(0, chang)
                    while rand==i:
                        rand = np.random.randint(0, chang)
                    pj=np.arange(chang)
                    np.random.shuffle(pj)
                    if self.pops[rand] is None:
                        continue
                    if chang<= self.popSize/5 :
                        self.pops[i][pj[0]] = self.pops[i][pj[0]] + (self.pops[rand][pj[0]] - self.pops[i][pj[0]]) * (1 + r1) * np.sin(2 * math.pi * r2)
                        self.pops[i][pj[1]] = self.pops[i][pj[1]] + (self.pops[rand][pj[1]] - self.pops[i][pj[1]]) * (
                                1 + r1) * np.cos(2 * math.pi * r2)
                    else:
                        for j in range(int(chang/2)-1):
                            self.pops[i][pj[2*j]] = self.pops[i][pj[2*j]] + (self.pops[rand][pj[2*j]] - self.pops[i][pj[2*j]]) * (
                                        1 + r1) * np.sin(2 * math.pi * r2)

                            self.pops[i][pj[2*j+1]] = self.pops[i][pj[2*j+1]] + (self.pops[rand][pj[2*j+1]] - self.pops[i][pj[2*j+1]]) * (
                                        1 + r1) * np.cos(2 * math.pi * r2)
                    pop_ = self.pops[i]
                    if pop_ is not None:
                        for j in range(len(pop_)):
                            if abs(pop_[j]) >= len(self.services[j]):
                                pop_[j] %= len(self.services[j])
                    pop_ = [int(x) for x in pop_]
                    self.popServices[i] = [self.services[j][pop_[j]] for j in range(len(pop_))]
                    violate, objFunc, _ = self.calc(self.popServices[i])
                    fitness = violate + objFunc
                    if self.bestFitness > fitness:
                        self.bestFitness = fitness
                        self.bestSolutions = self.popServices[i]
                        self.bestPops = self.pops[i]

                else:
                    r3 = np.random.random()
                    while r3 == 0:
                        r3 = np.random.random()
                    r4 = np.random.random()
                    while r4 == 0:
                        r4 = np.random.random()
                    C1 = 2 * r4 * (1 - t / self.MAX_Iter)

                    rand1 = np.random.randint(0, chang)

                    while rand1 == i:
                        rand1 = np.random.randint(0, chang)
                    beta = 3 / 2
                    delta = ((math.gamma(1 + beta) * np.sin((np.pi * beta) / 2))
                             / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
                    u = np.random.randn(chang)
                    v = np.random.randn(chang)
                    lf = 0.05 * ((u * delta) / (abs(v) ** (1 / beta)))
                    if self.pops[rand1] is None:
                        continue

                    pop_ = [(r3 * idx1 - r4 * idx2) + (idx3 -idx2) * C1 * lf_ for idx1,idx2,idx3,lf_ in zip(self.bestPops,self.pops[i],self.pops[rand1],lf)]

                if pop_ is not None:
                    for j in range(len(pop_)):
                        if abs(pop_[j]) >= len(self.services[j]):
                            pop_[j] %= len(self.services[j])
                pop_ = [int(x) for x in pop_]
                self.popServices[i] = [self.services[j][pop_[j]] for j in range(len(pop_))]
                violate, objFunc, _ = self.calc(self.popServices[i])
                fitness = violate + objFunc
                if self.bestFitness > fitness:
                    self.bestFitness = fitness
                    self.bestSolutions = self.popServices[i]
                    self.bestPops = self.pops[i]
                for i in range(len(self.pops[i])):
                    if self.pops[i] is None:
                        continue
                    if Bf[i] <= wf:
                        r5 = np.random.random()
                        r6 = np.random.random()
                        r7 = np.random.random()
                        while r5 == 0:
                            r5 = np.random.random()
                        while r6 == 0:
                            r6 = np.random.random()
                        while r7 == 0:
                            r7 = np.random.random()
                        c2 = 2 * wf * self.popSize
                        rand2 = np.random.randint(0, chang)
                        xstep=  chang * np.exp((-c2*t)/self.MAX_Iter)
                        if self.pops[rand2] is None:
                            continue
                        pop_=[r5 * idx4 -r6 * idx5 + r7 *xstep for idx4,idx5 in zip(self.pops[i],self.pops[rand2])]
                    if pop_ is not None:
                        for j in range(len(pop_)):
                            if abs(pop_[j]) >= len(self.services[j]):
                                pop_[j] %= len(self.services[j])
                    self.pops[i] = pop_
                    if pop_==None:
                        continue
                    pop_ = [int(x) for x in pop_]
                    self.popServices[i] = [self.services[j][pop_[j]] for j in range(len(pop_))]
                    violate, objFunc, _ = self.calc(self.popServices[i])
                    fitness = violate + objFunc
                    if self.bestFitness > fitness:
                        self.bestFitness = fitness
                        self.bestSolutions = self.popServices[i]
                        self.bestPops = self.pops[i]

            t += 1

            print(self.bestFitness)
            self.bestFitnesses.append(self.bestFitness)
        # print(self.bestFitnesses)

        return self.bestFitness, self.bestSolutions


class BWO:
    def __init__(self, dataset, serCategory, MLESWOAtest, ML2PNWOATest, MLWOATest, ESWOAtest, serviceNumber, reduct,
                 epoch, MAX_Iter, popSize):
        self.dataset = dataset + "/"
        self.serCategory = serCategory
        self.MLESWOAtest = MLESWOAtest
        self.ML2PNWOATest = ML2PNWOATest
        self.MLWOATest = MLWOATest
        self.ESWOAtest = ESWOAtest
        self.serviceNumber = serviceNumber
        self.reduct = reduct
        self.epoch = epoch
        self.MAX_Iter = MAX_Iter
        self.popSize = popSize

        self.times = 0
        self.qosNum = 4
        self.train = False
        self.sSetList = None

    def start(self):
        beat = 100
        if self.ML2PNWOATest:
            if self.epoch >= 0:
                with open(f"./solutions/PNHigh/{self.dataset}/allActions{self.epoch}.txt") as f:
                    allActions = json.load(f)
            else:
                with open(f"./solutions/pretrained/{self.dataset[:-1]}-PNHigh.txt") as f:
                    allActions = json.load(f)

            allActionsSolution = [[0] * self.serCategory for _ in range(1000)]
            for i in range(len(allActions)):
                for j in range(len(allActions[i])):
                    allActionsSolution[j][i] = allActions[i][j][: self.qosNum]

            newSolution = []
            self.sSetList = [set() for _ in range(len(allActionsSolution))]
            for i in range(len(allActionsSolution)):
                _newSolution = []
                for action in allActionsSolution[i]:
                    if sum(action[:]) != 3:
                        _newSolution.append(action)
                        _allAction = tuple([round(action[q], 5) for q in range(self.qosNum)])
                        self.sSetList[i].add(_allAction)
                newSolution.append(_newSolution)


        elif self.MLWOATest:
            print("MLWOATEXT")
            newSolution = []
            newServiceFeatures, newlabels = loadDataPN(self.epoch, dataset=self.dataset[:-1], serviceNumber=1)
            self.sSetList = [set() for _ in range(len(newServiceFeatures) // 4)]

            idx = 0
            for serviceFeatures in newServiceFeatures[len(newServiceFeatures) // 4 * 3:]:
                _newSolution = []
                for i in range(0, len(serviceFeatures)):
                    if sum(serviceFeatures[i][1: self.qosNum + 1]) != 3:
                        _newSolution.append(serviceFeatures[i][1: self.qosNum + 1])
                        _allAction = tuple([round(serviceFeatures[i][1 + q], 5) for q in range(self.qosNum)])
                        self.sSetList[idx].add(_allAction)

                newSolution.append(_newSolution)

                idx += 1


        else:
            if not self.train:
                newSolution = [None] * 1000
            else:
                newSolution = [None] * 4000

        newServiceFeatures, constraintsList, minCostList = loadDataOther(self.dataset, self.reduct,
                                                                         sSetList=self.sSetList, train=self.train)

        qualitiesInit = {
            "quality": [],
            "time": [],
            "averageQ": 0,
            "averageT": 0
        }

        # ML+ESWOA test
        if self.MLESWOAtest:
            newServiceFeatures, _ = loadDataPN(epoch=self.epoch, dataset=self.dataset[:-1],
                                               serviceNumber=self.serviceNumber)  # normal 2 qws 4
            serviceFeatures = []
            serviceCategories = []

            for k in range(len(newServiceFeatures)):  # 4000
                serviceCategory = []
                serviceFeature = []
                for i in range(len(newServiceFeatures[k]) // self.serviceNumber):
                    _serviceFeature = []
                    for j in range(self.serviceNumber):
                        feature = newServiceFeatures[k][i * self.serviceNumber + j][1: self.qosNum + 1]
                        if sum(feature[1:]) != 3:
                            _serviceFeature.append(tuple(feature))

                    if len(_serviceFeature) > 0:
                        serviceFeature.append(_serviceFeature)
                        serviceCategory.append(i)

                serviceCategory = set(serviceCategory)
                serviceCategories.append(serviceCategory)
                serviceFeatures.append(serviceFeature)

            if self.train:
                newServiceFeatures = serviceFeatures
            else:
                newServiceFeatures = serviceFeatures[len(minCostList) // 4 * 3:]

        bestFitnesses = [[] for _ in range(self.MAX_Iter)]

        if self.train:
            _min = 0
        else:
            _min = len(minCostList) // 4 * 3  # 3000

        for newServiceFeature, constraints, minCost, solution, idx in zip(newServiceFeatures, constraintsList,
                                                                          minCostList[_min:], newSolution,
                                                                          range(_min, len(minCostList))):

            t = time.time()
            '''if len(newServiceFeature)==20:#3013 3040 3041 3075
                print(idx)
            continue
            exit()'''
            #if idx == 3003:
            if not solution:
                model = BWO_(newServiceFeature, constraints, popSize=self.popSize, MAX_Iter=self.MAX_Iter,
                             y='buyuan')
            else:
                model = BWO_(newServiceFeature, constraints, solution, popSize=self.popSize,
                             MAX_Iter=self.MAX_Iter)
            q, sol = model.start()


            for i in range(self.MAX_Iter):
                bestFitnesses[i].append(model.bestFitnesses[i])

            print("第t次", self.times, beat)
            tt = time.time() - t

            qualitiesInit["quality"].append(minCost / q)
            qualitiesInit["time"].append(tt)
            qualitiesInit["averageQ"] = sum(qualitiesInit["quality"]) / (self.times + 1)
            qualitiesInit["averageT"] = sum(qualitiesInit["time"]) / (self.times + 1)
            print(idx, qualitiesInit["averageQ"], qualitiesInit["averageT"])
            self.times += 1

            '''if self.ML2PNWOATest:
                with open(f"./solutions/WOA/{self.dataset}/ML+2PN+WOA.txt", "w") as f:
                    json.dump(qualitiesInit, f)

            if self.ESWOAtest:
                url = f"./solutions/WOA/{self.dataset}/ESWOA.txt"
                with open(url, "w") as f:
                    json.dump(qualitiesInit, f)

            if self.MLESWOAtest:
                url = f"./solutions/WOA/{self.dataset}/ML+ESWOA.txt"
                with open(url, "w") as f:
                    json.dump(qualitiesInit, f)'''

