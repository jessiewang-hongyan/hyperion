from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import List
import random

# scores: 
class MergeResult(ABC):
    @abstractmethod
    def merge(self, scores: List[Tensor]):
        pass

# basic mering step, assuming 2 clfs
# when 2 clfs agrees, use the agreed result
# else, choose the result with higher score
# if both labels have the same score, randomly chose one with equal weights
class BasicMergeResult(MergeResult):
    def merge(self, scores: List[Tensor]):
        score0 = scores[0]
        score1 = scores[1]
        merged = []

        results0 = 1 - torch.argmax(score0, dim=0)
        results1 = torch.argmax(score1, dim=0)
        results0 = results0.tolist()
        results1 = results1.tolist()

        print(f'results0: {results0}')
        print(f'results1: {results1}')

        for idx, (r0, r1) in enumerate(zip(results0, results1)):
            if r0 == r1:
                merged.append(r0)
            else:
                s0 = score0[1 - r0][idx]
                s1 = score1[r1][idx]

                print(f'idx:{idx}, s0={s0}, s1={s1}')

                if s0 == s1:
                    merged.append(random.choice([r0, r1]))
                elif s0 > s1:
                    merged.append(r0)
                else:
                    merged.append(r1)
        return merged

if __name__ == "__main__": 

    # 4 cases:
    # 1) r0==r1-------------------0
    # 2) r0!=r1, s0>s1------------0
    # 3) r0!=r1, s0<s1------------0
    # 4) r0!=r1, s0==s1-----------random
    scores=[
        torch.Tensor([[0.1, 0.1, 0.7, 0.7], [0.9, 0.9, 0.3, 0.3]]), # r0: 1-[1, 1, 0, 0] = [0, 0, 1, 1]
        torch.Tensor([[0.9, 0.2, 0.9, 0.7], [0.1, 0.8, 0.1, 0.3]]) # r1: [0, 1, 0, 0]
    ]

    B = BasicMergeResult()
    print(B.merge(scores))