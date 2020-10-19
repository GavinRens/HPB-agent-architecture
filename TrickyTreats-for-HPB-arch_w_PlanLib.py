# -*- coding: utf-8 -*-
"""
Created on 13 October 2016

@author: Gavin Rens
"""


import math
import random
import time





# A plain online POMDP planner, possibly with the simple Reduce by Mean optimization
# :main: A M by M grid-world in which items must be collected. The collecting robot can move in the four directions. There is at most one item per cell. There are M*N or fewer items randomly scattered around.


# Choose problem size: TrickyTreats[M,N]
# M: grid length and breadth
# N: # treats


M = 6  # M >= 4

N = math.ceil((M**2)/3)   # N <= M**2 - 1

# The set of goals
GOALS = ['11', '1M', 'M1', 'MM', 'collect']

# The set of goals which should not be pursued simultaneously
DISJOINT = ['11', '1M', 'M1', 'MM']

# The weights of goals
# Note that setting the weight of ONE corner to 1 results in poor performance.
weight = list()
weight.append(['11', 0.5/4])
weight.append(['1M', 0.5/4])
weight.append(['M1', 0.5/4])
weight.append(['MM', 0.5/4])
weight.append(['collect', 0.5])
#0.0625
#0.1875
# Declare the reward discount factor
gamma = 0.9
#gamma = 1

# Stochasticity factor:
# SF=0 # most uncertain in action and perception
# SF=1 # most certain in action and perception
SF = 0.95

# Let x be the nuof correct states the agent should be in after executing some action, and let y be the nuof incorrect states the agent may  up in.
# do, in general, the prob of ending up in a correct state is 1/(x + (1 - SF)*y) and the prob of ing up in an incorrect state is (1 - SF)/(x + (1 - SF)*y)

#   Change/choose the parameters in experiment( ) at the  of this file
#   Change/choose the desired algo in def Sim( ) close to the  of the file

# Satisfaction Change Threshold
SCT = 0.05

# Number of most recent satf levels to use to calculate the rate of Change
MEMORY = 5

# Thresholds for deciding whether two intention sets, resp., belief states are similar (enough)
BSI_threshold = 0.9
SBS_threshold = 0.9


# Utility defs ###############################################


def printList(List):
    for i in List:
        print(i)
    
    
# Takes a string Str as index and returns the assoc. element of List
def si(List,Str):
    for j in List:
        if j[0] == Str:
            return j[1]
        

# Takes a string Str as index and assigns Val to the corresponding element of List
def sia(List,Str,Val):
    for j in List:
        if j[0] == Str:
            j[1] = Val
        

# Print a belief state
def printBS(b):
    for sp in b:
        printState(sp[0]); print(sp[1])


# Print a state
def printState(s):
    print(s[0],s[1],s[2],s[3])


# Print all treats not yet eaten
def printTreats():
    print("Treat here:")
    for x in range(1, M + 1):
        for y in range(1, M + 1):
            if treatHere(x, y): print(x, y)


# Are two states the same?
def sameStates(s1, s2):
    for f,g in zip(s1,s2):
        if f != g:
            return False
    return True


# Shuffle the elements of List
def shuffle(List):
    Shuffed = []
    for i in range(1, len(List)):
        Shuffed.append(List[i])
    Shuffed.append(List[0])
    return Shuffed


# Determine which cells are surrounding the robot, assuming it is certain of its location
def cellsAround(x,y):
    if x == 1:
        if y == 1: return [[2,1],[2,2],[1,2]]
        if y == M: return [[2,y],[y-1,y-1],[1,y-1]]
        return [[x,y-1],[x+1,y-1],[x+1,y],[x+1,y+1],[x,y+1]]
    if x == M:
        if y == 1: return [[x-1,1],[x-1,2],[x,2]]
        if y == M: return [[x-1,y],[x-1,y-1],[x,y-1]]
        return [[x,y-1],[x-1,y-1],[x-1,y],[x-1,y+1],[x,y+1]]
    if y == 1:  # but the robot won't be in corners here
        return [[x-1,y],[x-1,y+1],[x,y+1],[x+1,y+1],[x+1,y]]
    if y == M:  # but the robot won't be in corners here
        return [[x-1,y],[x-1,y-1],[x,y-1],[x+1,y-1],[x+1,y]]
    return [[x-1,y-1],[x-1,y],[x-1,y+1],[x,y-1],[x,y+1],[x+1,y-1],[x+1,y],[x+1,y+1]]


def treatHere(x,y):
    global Treats
    for t in Treats:
        if t[0] == x and t[1] == y:
            return True
    return False


def removeTreatHere(x,y):
    global Treats
    tmp = []
    for t in Treats:
        if not(t[0] == x and t[1] == y):
            tmp.append(t)
    Treats = tmp


def addTreatHere(x,y):
    global Treats
    Treats.append([x,y])


def mostIntense(DesireLevels):
    maxVal = - 10**30  #math.huge
    most = None
    for d in DesireLevels:
        if d[1] > maxVal:
            maxVal = d[1]; most = d[0]
    return most


def rateOfSatf(g,w,sl):
    sum = 0
    for i in range(len(si(sl,g)) - w + 2, len(si(sl,g))):
        sum += si(sl,g)[i] - si(sl,g)[i-1]
    return float(sum)/(len(si(sl,g))-1)


def remoov(g,C,threshold,sl):
    if len(si(sl,g)) < 2: return False
    if C < threshold: return True
    else: return False


def removeIntention(g,I):
    Inew = []
    for gg in I:
        if g != gg: Inew.append(gg)
    return Inew


def IN_DISJOINT(g):
    for d in DISJOINT:
        if g == d: return True
    return False


def MeanAsThreshold(b):
    if len(b) > 2:
        b_reduc = []
        mean = 1.0/len(b)
        for sp in b:
            if sp[1] >= mean: b_reduc.append(sp)
        return normalize(b_reduc)
    else: return b


# Directed Divergence
def R(c,b):
    for spc in c:
        spc_isin_spb = False
        for spb in b:
            if sameStates(spb[0], spc[0]):
                spc_isin_spb = True
        if spc_isin_spb is False:
            return 5.0
    # If this point is reached, for every state mentioned in c, that state is also mentioned in b
    # That is, the following holds: for all states s in S, if c(s)>0, then b(s)>0
    # Then the directed divergence of c from b is defined -- and computed as:
    sum = 0
    for spb in b:
        for spc in c:
            if sameStates(spb[0], spc[0]):
                sum += spc[1] * math.log(spc[1] / spb[1])
    return sum


# Similarity between two belief states (different measure to directed divergence)
# The reason for the complicated code is due to zero-prob states not being mentioned in belief states
def SBS(c,b):
    sum = 0
    for spc in c:
        spc_isin_spb = False
        for spb in b:
            if sameStates(spb[0], spc[0]):
                spc_isin_spb = True
                sum += math.fabs(spb[1] - spc[1])
        if not spc_isin_spb:
            sum += spc[1]
    for spb in b:
        spb_isin_spc = False
        for spc in c:
            if sameStates(spb[0], spc[0]):
                spb_isin_spc = True
        if not spb_isin_spc:
            sum += spb[1]
    return 1-sum/2


# Believed similarity bwix intention sets I1 and I2
def BSI(I1, I2, B1, B2):
    unionSet = set()
    for g in I1:
        unionSet.add(g)
    for g in I2:
        unionSet.add(g)
    I1Set = set(I1)
    I2Set = set(I2)
    intersectionSet = I1Set.intersection(I2Set)
    sum = 0
    for g in intersectionSet:
        sum += 1 - math.fabs(SatLev_B(g, B1) - SatLev_B(g, B2))
    return sum/len(unionSet)








# The POMDP #####################################################-

#POMDP = <S,A,T,O,P,R,C> # states, actions, transition def, observations, perception def, reward def, cost def


# The states
# a state is quadtruple: x-coord, y-coord, direction facing, treat present
States=[]

for x in range(1,M+1):
    for y in range(1,M+1):
        States.append([x,y,'n',0,])
        States.append([x,y,'e',0])
        States.append([x,y,'w',0])
        States.append([x,y,'s',0])
        States.append([x,y,'n',1])
        States.append([x,y,'e',1])
        States.append([x,y,'w',1])
        States.append([x,y,'s',1])




# The actions
Actions = ['l','r','f','e','s']  # turn left, turn right, move one cell forward, eat, sniff treat



# The observations
Observations = []
Observations.append(['l',['obsNil']])
Observations.append(['r',['obsNil']])
Observations.append(['f',['obsNil']])
Observations.append(['e',['obsNil']])
Observations.append(['s',[0,1]])



# The transition function
def T(sat,a,sto):  # sat: state at, a: action, sto: state to which going
    locationAtX=sat[0]; locationAtY=sat[1]; sat_dir=sat[2]; sat_treat=sat[3]
    locationToX=sto[0]; locationToY=sto[1]; sto_dir=sto[2]; sto_treat=sto[3]

    if a == 'l': # (1-SF)/2 + SF + (1-SF)/2 = 1
        if locationAtX == locationToX and locationAtY == locationToY and sat_treat == sto_treat:
            if sat_dir=='n':
                if sto_dir=='n' : return (1-SF)/2
                if sto_dir=='w' : return SF
                if sto_dir=='s' : return (1-SF)/2
            
            if sat_dir=='e':
                if sto_dir=='e' : return (1-SF)/2
                if sto_dir=='n' : return SF
                if sto_dir=='w' : return (1-SF)/2
            
            if sat_dir=='w':
                if sto_dir=='w' : return (1-SF)/2
                if sto_dir=='s' : return SF
                if sto_dir=='e' : return (1-SF)/2
            
            if sat_dir=='s':
                if sto_dir=='s' : return (1-SF)/2
                if sto_dir=='e' : return SF
                if sto_dir=='n' : return (1-SF)/2
            
    if a == 'r': # (1-SF)/2 + SF + (1-SF)/2 = 1
        if locationAtX == locationToX and locationAtY == locationToY and sat_treat == sto_treat :
            if sat_dir=='n' :
                if sto_dir=='n' : return (1-SF)/2
                if sto_dir=='e' : return SF
                if sto_dir=='s' : return (1-SF)/2
            
            if sat_dir=='e' :
                if sto_dir=='e' : return (1-SF)/2
                if sto_dir=='s' : return SF
                if sto_dir=='w' : return (1-SF)/2
            
            if sat_dir=='w' :
                if sto_dir=='w' : return (1-SF)/2
                if sto_dir=='n' : return SF
                if sto_dir=='e' : return (1-SF)/2
            
            if sat_dir=='s' :
                if sto_dir=='s' : return (1-SF)/2
                if sto_dir=='w' : return SF
                if sto_dir=='n' : return (1-SF)/2
            
    if a == 'f':    # 2 * SF/2 + 1-SF = 1. It is "2 * SF/2" because there are two ways in which the agent can land in locationToY == locationAtY+1:
                    # where sto_treat == 0 and where sto_treat == 1
                    # If the agent does not move (locationToY == locationAtY), then the 'treat-status' may not change (and sat_treat == sto_treat).
        if sat_dir=='n' and sto_dir=='n' and locationToX == locationAtX and locationAtY != M :
            if locationToY == locationAtY+1 : return SF/2
            if locationToY == locationAtY and sat_treat == sto_treat: return 1-SF
        
        if sat_dir=='e' and sto_dir=='e' and locationToY == locationAtY and locationAtX != M :
            if locationToX == locationAtX+1 : return SF/2
            if locationToX == locationAtX and sat_treat == sto_treat: return 1-SF
        
        if sat_dir=='w' and sto_dir=='w' and locationToY == locationAtY and locationAtX != 1 :
            if locationToX == locationAtX-1 : return SF/2
            if locationToX == locationAtX and sat_treat == sto_treat: return 1-SF
        
        if sat_dir=='s' and sto_dir=='s' and locationToX == locationAtX and locationAtY != 1 :
            if locationToY == locationAtY-1 : return SF/2
            if locationToY == locationAtY and sat_treat == sto_treat: return 1-SF
        
        # When the robot is on the border, facing the border
        if sameStates(sat, sto) :
            if sat_dir=='n' and locationAtY == M : return 1.0
            if sat_dir=='e' and locationAtX == M : return 1.0
            if sat_dir=='w' and locationAtX == 1 : return 1.0
            if sat_dir=='s' and locationAtY == 1 : return 1.0

    if a == 'e':
        if locationAtX == locationToX and locationAtY == locationToY and sat_dir == sto_dir and sto_treat == 0 : # and sat_treat == 1
        # whether or not there is a treat at the current location, the eating action is possible
            return 1.0
        
    if a == 's': # the agent either smells a treat or smells no treat; smelling is deterministic
        if sameStates(sat, sto): return 1.0
    
    return 0.0 # for all cases not considered; they are not possible




# The perception function
def P(sat, a, e):  #  sat: a state, a: action, e: observation
    global Treats
    if (a == 'l' or a == 'r' or a == 'f' or a == 'e') and e == 'obsNil': return 1.0
    if a == 's':
        if e == sat[3]: return SF
        else: return 1-SF
    return 0.0 # for all other cases


















# The beleif update procedure ############################################

# To normalize a belief state; used in SE() below
def normalize(b):
    B_norm = []
    sum = 0
    for sp in b:
        sum = sum + sp[1]
    for sp in b:
        p_new = float(sp[1])/sum
        B_norm.append([sp[0], p_new])
    return B_norm



# The state estimation (belief update) def
def SE(a,e,b_c):
    B_n = []
    for sto in States:  # for every possible state; States is global
        sum = 0
        p_sto = P(sto, a, e)  # prob. of perceiving e in new state, given a
        if p_sto != 0:
            for sat in b_c:  # for every state in the current belief state
                sum += T(sat[0], a, sto) * sat[1]  # expected prob. of reaching new state sto
        if sum != 0:  # only add (state,prob.) pair that are possible
            B_n.append([sto, p_sto*sum])
    return normalize(B_n)



# The probability of reaching the belief state, given a was performed in b and e is perceived
def Prob_ofnew_BS(a,e,b):
    sum1 = 0
    for sto in States: # for every possible state
        sum2 = 0
        for sat in b :    # for every state in the current belief state
            sum2 += T(sat[0], a, sto) * sat[1]  # expected prob. of reaching new state sto
        sum1 += P(sto, a, e) * sum2
    return sum1
    

# Update the robot's current belief-state according to the passive location sensor.
def passiveLocationSensing(b_c):
    if SF == 1: return b_c
    x = currentState[0]
    y = currentState[1]
    # When the agent senses its location, it senses the cells around its actual location (according to currentState)
    CellsAround = cellsAround(x,y)
    b_n = []
    p_incorrect_loc = float((1-SF)/len(CellsAround))
    p_correct_loc = SF
    for sat in b_c: # for every state in the current belief-state
        for loc in CellsAround: # for all cells around the actual current cell
            if sat[0][0] == loc[0] and sat[0][1] == loc[1]:
                b_n.append([sat[0], sat[1]*p_incorrect_loc])
        if sat[0][0] == x and sat[0][1] == y:
                b_n.append([sat[0], sat[1]*p_correct_loc])
    if len(b_n) == 0:
        for loc in CellsAround: # for all cells around the actual current cell
            b_n.append([[loc[0],loc[1],'n',0], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'n',1], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'e',0], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'e',1], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'w',0], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'w',1], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'s',0], p_incorrect_loc/8])
            b_n.append([[loc[0],loc[1],'s',1], p_incorrect_loc/8])
        b_n.append([[x,y,'n',0], p_correct_loc/8])
        b_n.append([[x,y,'n',1], p_correct_loc/8])
        b_n.append([[x,y,'e',0], p_correct_loc/8])
        b_n.append([[x,y,'e',1], p_correct_loc/8])
        b_n.append([[x,y,'w',0], p_correct_loc/8])
        b_n.append([[x,y,'w',1], p_correct_loc/8])
        b_n.append([[x,y,'s',0], p_correct_loc/8])
        b_n.append([[x,y,'s',1], p_correct_loc/8])
    return normalize(b_n)














def Cost(a,s):  # a: action, s: state
    if a == 's': return -10
    else: return 0
    

# The average cost of performing a in b
def Cost_B(a,b):
    sum = 0
    for sp in b:
        sum += Cost(a,sp[0]) * sp[1]
    return sum


def Satf(g, a, s):
    global Treats
    x1 = s[0]
    y1 = s[1]
    # x2;  y2
    if g != 'collect':
        if g == '11':
            x2 = 1
            y2 = 1
        if g == '1M':
            x2 = 1
            y2 = M
        if g == 'M1':
            x2 = M
            y2 = 1
        if g == 'MM':
            x2 = M
            y2 = M
        dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
        return 1 - float(dist) / (2 * (M - 1))
    else:  # g is 'collect'
        dist = M ** 2  # a large number
        eatUtil = 0
        x = 5
        for T in Treats:
            x2 = T[0]
            y2 = T[1]
            if math.fabs(x1 - x2) + math.fabs(y1 - y2) < dist:
                dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
        if a == 'e':
            if s[3] == 1:
                eatUtil = x
            else:
                eatUtil = 0
        return (1 - float(dist) / (2 * (M - 1)) + eatUtil + x) / (1 + x)


def SatLev(g, s):
    global Treats
    x1 = s[0]
    y1 = s[1]
    # x2;  y2
    if g != 'collect':
        if g == '11':
            x2 = 1
            y2 = 1
        if g == '1M':
            x2 = 1
            y2 = M
        if g == 'M1':
            x2 = M
            y2 = 1
        if g == 'MM':
            x2 = M
            y2 = M
        dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
        return 1 - float(dist) / (2 * (M - 1))
    else:  # g is 'collect'
        dist = M ** 2  # a large number
        x = 5
        for T in Treats:
            x2 = T[0]
            y2 = T[1]
            if math.fabs(x1 - x2) + math.fabs(y1 - y2) < dist:
                dist = math.fabs(x1 - x2) + math.fabs(y1 - y2)
        if s[3] == 0:
            eatUtil = x
        else:
            eatUtil = -x
        return (1 - float(dist) / (2 * (M - 1)) + eatUtil + x) / (1 + 2*x)


# The average satisfaction for doing a in b while the goal is g
def Satf_B(g, a, b):
    sum = 0
    for sp in b:
        sum += Satf(g,a,sp[0]) * sp[1]
    return sum


# The average satisfaction for being in b while the goal is g
def SatLev_B(g, b):
    sum = 0
    for sp in b:
        sum += SatLev(g, sp[0]) * sp[1]
    return sum


def i(g,I):
    for el in I:
        if el == g: return 1
    return 0














# The action selector with Mean as Threshold condensation method #################################-

def PolicyGenerator(b, I, h):  # b: a belief state, h: planning horizon (# steps sought), p: policy
    # print('size of Bel stt:', len(b))
    if h == 0: return 'stop', 0
    if h > 0:
        #bestAction = None
        maxVal = -10 ** 30
        global Actions
        Actions = shuffle(Actions)
        for a in Actions:
            policy, value = PG_AUX(a, b, I, h)
            if value > maxVal:
                maxVal = value
                #bestAction = a
                bestPolicy = policy
        return bestPolicy, maxVal


def PG_AUX(a, b, I, h):
    sum = 0
    policy = []
    global Observations
    for e in si(Observations, a):
        prob = Prob_ofnew_BS(a, e, b)
        if prob > 0:
            b_new = SE(a, e, b)
            sub_policy, V_star = PolicyGenerator(MeanAsThreshold(b_new), I, h - 1)
            # policy, V_star = PolicyGenerator(b_new, I, h-1)
            sum += prob * V_star
            policy.append([e, sub_policy])
        return [a, policy], i('11', I) * si(weight, '11') * Satf_B('11', a, b) + i('1M', I) * si(weight, '1M') * Satf_B('1M', a, b) + \
               i('M1', I) * si(weight, 'M1') * Satf_B('M1', a, b) + i('MM', I) * si(weight, 'MM') * Satf_B('MM', a, b) + \
               i('collect', I) * si(weight, 'collect') * Satf_B('collect', a, b) + (gamma * sum)










def updateState(a):
    rand = random.random()
    if a == 'l' :
        if currentState[2]=='n':
            if rand <= (1-SF)/2: currentState[2]='n'
            elif rand <= 1-SF: currentState[2]='s'
            else: currentState[2]='w' 
        elif currentState[2]=='e':
            if rand <= (1-SF)/2: currentState[2]='e'
            elif rand <= 1-SF: currentState[2]='w'
            else: currentState[2]='n' 
        elif currentState[2]=='w':
            if rand <= (1-SF)/2: currentState[2]='w'
            elif rand <= 1-SF: currentState[2]='e'
            else: currentState[2]='s' 
        elif currentState[2]=='s':
            if rand <= (1-SF)/2: currentState[2]='s'
            elif rand <= 1-SF: currentState[2]='n'
            else: currentState[2]='e' 
    if a == 'r' :
        if currentState[2]=='n':
            if rand <= (1-SF)/2: currentState[2]='n'
            elif rand <= 1-SF: currentState[2]='s'
            else: currentState[2]='e' 
        elif currentState[2]=='e':
            if rand <= (1-SF)/2: currentState[2]='e'
            elif rand <= 1-SF: currentState[2]='w'
            else: currentState[2]='s' 
        elif currentState[2]=='w':
            if rand <= (1-SF)/2: currentState[2]='w'
            elif rand <= 1-SF: currentState[2]='e'
            else: currentState[2]='n' 
        elif currentState[2]=='s':
            if rand <= (1-SF)/2: currentState[2]='s'
            elif rand <= 1-SF: currentState[2]='n'
            else: currentState[2]='w' 
    if a == 'f' :
        if currentState[2]=='n':
            if (currentState[1] != M):
                if rand <= SF: currentState[1] += 1
        if currentState[2]=='e':
            if (currentState[0] != M):
                if rand <= SF: currentState[0]+=1
        if currentState[2]=='w':
            if (currentState[0] != 1):
                if rand <= SF: currentState[0]-=1
        if currentState[2]=='s':
            if (currentState[1] != 1):
                if rand <= SF: currentState[1]-=1
        if treatHere(currentState[0],currentState[1]): currentState[3] = 1
        else: currentState[3] = 0
    if a == 'e':
        if treatHere(currentState[0],currentState[1]):
            removeTreatHere(currentState[0],currentState[1])
            currentState[3] = 0
    #if a == 's' # : nothing


def getObservation(sat,a):
    random.seed()
    rand = random.random()
    if a == 'l' or a == 'r' or a == 'f' or a == 'e': return 'obsNil'
    if a == 's':
        if rand <= SF:
            return sat[3]
        elif sat[3] == 1: return 0
        else: return 1
        
    
def execute(a, I):
    global currentBS
    total_value = 0
    print('act:',a)
    reward = i('11', I) * si(weight, '11') * Satf('11', a, currentState) + i('1M', I) * si(weight, '1M') * Satf('1M', a, currentState) + i('M1', I) * si(weight, 'M1') * Satf('M1', a, currentState) + i('MM', I) * si(weight, 'MM') * Satf('MM', a, currentState) + i('collect', I) * si(weight, 'collect') * Satf('collect', a, currentState)
    #print('reward:',reward)
    updateState(a)  # updates the current state (execute the action)
    e = getObservation(currentState,a)
    currentBS = SE(a,e,currentBS)
    #print('After BU:');printBS(currentBS)
    print('After SU:');printState(currentState)
    return reward, e







################################### THE SIMULATOR #######################################

def Sim(nuof_acts, h):
    global currentState
    global currentBS
    global TotalTreatsEaten
    global TotalCornersVisited
    global Treats
    global One_one
    global One_six
    global Six_one
    global Six_six
    global TotalPlanLibSizes
    global TotalTimesRetrieved

    global MEMORY
    global BSI_threshold
    global SBS_threshold
    step = 0
    rewards = 0
    # Initial desire levels
    DesireLevels = [['11', 0], ['1M', 0], ['M1', 0], ['MM', 0], ['collect', 0]]
    # Define the list of satisfaction levels (initially empty)
    Satf_levels = []
    # Initial sequence of satisfaction levels
    Satf_levels.append(['11', []])
    Satf_levels.append(['1M', []])
    Satf_levels.append(['M1', []])
    Satf_levels.append(['MM', []])
    Satf_levels.append(['collect', []])
    I = []
    # Initially, the set of intentions is the set of goals
    #for g in GOALS:
    #    I.append(g)
    # Initialize the empty plan library
    PlanLib = []
    prevX = 0
    prevY = 0
    # printTable(I)

    Init_bel_stt()
    #print 'agents init state:'
    #printBS(currentBS)
    treat_here = treatHere(currentBS[0][0][0], currentBS[0][0][1])
    if treat_here: th = 1
    else: th = 0
    currentState = [currentBS[0][0][0], currentBS[0][0][1], currentBS[0][0][2], th]
    nuof_acts_sofar = 0
    while nuof_acts_sofar < nuof_acts:
        retrieved_policy = False
        for plan in PlanLib:
            I_lib = plan[0]
            B_lib = plan[1]
            if BSI(I_lib, I, B_lib, currentBS) >= BSI_threshold and SBS(B_lib, currentBS) >= SBS_threshold:
                print('Retrieving:', BSI(I_lib, I, B_lib, currentBS), SBS(B_lib, currentBS))
                policy = list(plan[2])
                retrieved_policy = True
                TotalTimesRetrieved += 1
                break
        if not retrieved_policy:
            policy, _ = PolicyGenerator(currentBS, I, h)
            PlanLib.append([I,currentBS,policy])
        polExecStep = 0
        #while policy != 'stop':
        while polExecStep < h/2: # at most, half of a full policy is executed; exec'n of deeper policy results in suboptimal actions
            step = step + 1
            print('step:', step)
            #print 'policy:', policy

            # Update satisfaction levels
            for g in I:
                for j in Satf_levels:
                    if j[0] == g:
                        j[1].append(SatLev_B(g, currentBS))

            # Add one goal at most?
            MI = mostIntense(DesireLevels)
            IN_I = False
            DIS_IN_I = False
            for g in I:
                if MI == g: IN_I = True
                if IN_DISJOINT(g) and g != MI: DIS_IN_I = True
            if not IN_I and not (IN_DISJOINT(MI) and DIS_IN_I): sia(Satf_levels, MI, []); I.append(MI)

            # Remove one goal at most?
            if len(I) > 1:
                justRemoved = 'nothing'
                for g in I:
                    if justRemoved == 'nothing' and len(si(Satf_levels, g)) >= MEMORY and remoov(g,rateOfSatf(g, MEMORY, Satf_levels), SCT, Satf_levels):
                        I = removeIntention(g, I)
                        justRemoved = g
                        print(g, 'removed')


            # Remove, special case: when the only intention must be disjoint from the most intense goal, and the intention should be removed
            # This case is not dealt with by previous removal procedure, because there, it is illegal to remove the last intention
            elif IN_DISJOINT(MI):
                for g in I:
                    if IN_DISJOINT(g) and g != MI and len(si(Satf_levels, g)) >= MEMORY and remoov(g, rateOfSatf(g, MEMORY, Satf_levels), SCT, Satf_levels):
                        print(len(si(Satf_levels, g)))
                        I = removeIntention(g, I)
                        sia(Satf_levels, MI, [])
                        I.append(MI)
                        print(g, 'removed')

            print('intentons:')
            for g in I:
                print(g)

            act = policy[0]
            sub_policy = policy[1]

            for T in Treats:
                if currentState[0] == T[0] and currentState[1] == T[1] and act == 'e':
                    TotalTreatsEaten += 1

            # Execute and update current belief-state
            # must update currentLocX, currentLocY, currentDir and currentBS after every action and observation
            reward, observation = execute(act, I)
            polExecStep += 1

            for p in sub_policy:
                if p[0] == observation:
                    policy = p[1]

            # record nuof times a corner is reached
            # but don't record it for consecutive steps
            if currentState[0] == 1 and currentState[1] == 1 and not (prevX == 1 and prevY == 1): One_one = One_one + 1
            if currentState[0] == 1 and currentState[1] == 6 and not (prevX == 1 and prevY == 6): One_six = One_six + 1
            if currentState[0] == 6 and currentState[1] == 1 and not (prevX == 6 and prevY == 1): Six_one = Six_one + 1
            if currentState[0] == 6 and currentState[1] == 6 and not (prevX == 6 and prevY == 6): Six_six = Six_six + 1
            prevX = currentState[0]
            prevY = currentState[1]

            currentBS = passiveLocationSensing(currentBS)
            currentBS = MeanAsThreshold(currentBS)

            # Update desire levels
            for d in DesireLevels:
                if i(d[0], I) == 0:
                    d[1] = d[1] + si(weight, d[0]) * (1 - SatLev_B(d[0], currentBS))

            nuof_acts_sofar += 1
            rewards += reward
            break # When uncommented, only first action of every policy is executed
            print

    TotalPlanLibSizes += len(PlanLib)

    return rewards

#################################### FIN SIMULATOR ####################################














def Init_bel_stt():
    global currentBS
    X = 0; Y = 0; D = ' '
    for x in range(1,M+1):
        rand = random.random()
        if rand <= float(x)/M: X = x; break
    for y in range(1,M+1):
        rand = random.random()
        if rand <= float(y)/M: Y = y; break
    for d in range(4):
        rand = random.random()
        if rand <= float(d+1)/4:
            if d == 1: D = 'n'  #; print('dir: n')
            elif d == 2: D = 'e'  #; print('dir: e')
            elif d == 3: D = 'w'  #; print('dir: w')
            else: D = 's'; break  #; print('dir: s')
    currentBS = [[[X, Y, D, 0], 0.5], [[X, Y, D, 1], 0.5]]


def Place_Treats():
    global Treats
    Treats = []
    num_added = 0  # nuof treats added
    while num_added < N:  # not all treats have been added
        k = 0  # counts the nuof cells considered for one placement
        for x in range(1,M+1):
            for y in range(1,M+1):
                rand = random.random()
                k += 1
                if (not treatHere(x,y)) and rand <= float(k)/(M**2) and num_added < N:
                    # as the next cell is considered, the probability that a treat will be added there increases
                    addTreatHere(x,y)
                    num_added += 1


Treats = []  # Treats[i][0] is the x coord of treat i; Treats[i][1] is the y coord of treat i
currentState = []
currentBS = []
TotalTreatsEaten = 0
One_one = 0
One_six = 0
Six_one = 0
Six_six = 0
TotalPlanLibSizes = 0
TotalTimesRetrieved = 0


def experiment(trials, nuof_acts, h):
    global TotalTreatsEaten
    rewards = 0
    totalstarttime = time.clock()

    all_rewards = [];  all_times = []
    random.seed()

    for j in range(trials):
        Place_Treats()
        #printTreats()
        print('trial:',j+1)
        start_time = time.clock()
        rewards = Sim(nuof_acts, h)
        fin_time = time.clock()
        all_rewards.append(rewards)
        all_times.append(fin_time - start_time)

    avg_treats_eaten = float(TotalTreatsEaten)/trials

    totalendtime = time.clock()
    Total_exp_tim = totalendtime-totalstarttime

    sum = 0.0

    for i in all_rewards: sum += i
    avg_rewards = sum/len(all_rewards)
    rewards_per_act = avg_rewards/nuof_acts

    sum = 0.0
    for i in all_times: sum = sum + i
    avg_times = sum/len(all_times)
    sec_per_act = avg_times/nuof_acts

    avg_rewards_per_second = avg_rewards/avg_times

    avg_planlib_size = float(TotalPlanLibSizes)/trials

    avg_times_retrieved = float(TotalTimesRetrieved)/trials

    print('experiment(',trials, nuof_acts, h,')')

    print('Treats eaten:',avg_treats_eaten)

    print('11: ', float(One_one) / trials, '16: ', float(One_six) / trials, '61: ', float(Six_one) / trials, '66: ', float(Six_six) / trials)

    print('PlanLib size at end of trial:', avg_planlib_size)

    print('Nuof times plan retrieved:', avg_times_retrieved)

    print('Rewards per act:',rewards_per_act)

    print('Secs per act:',sec_per_act)

    print('Rewards per second:',avg_rewards_per_second)

    print('Avg duration of trial:',Total_exp_tim / trials)





#   Change these parameters as desired.
#       trials: the nuof trials
#       nuof_acts: the nuof plan-act iterations per trial
#       h: horizon
#   experiment(trials,  nuof_acts,  h)

#experiment(30, 100, 4)
experiment(3, 100, 3)

#print SBS([[['a'],0.333],[['b'],0.333],[['c'],0.333]], [[['a'],0.333],[['b'],0.333],[['c'],0.233],[['d'],0.1]])