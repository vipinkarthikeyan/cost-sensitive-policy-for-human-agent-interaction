# Cost-Sensitive Clarification Under Uncertain Human Intent

## Overview

**When should a robot act immediately, and when should it ask the human for clarification?**

In many real-world settings, human instructions are ambiguous. For example, if a user says:

- "get me the mug"

and there are two mugs on the table, the robot does not know exactly which object is intended. Acting too quickly may lead to mistakes. Asking too often may make the interaction slow and annoying.

This project builds a decision-making framework for that tradeoff.

The overall pipeline is:

**Human instruction -> language grounding -> belief over possible goals -> decision policy -> action**

In our implementation:
- the human types a natural-language instruction
- a language model assigns a probability distribution over candidate objects
- the robot uses that probability distribution to decide whether to:
  - **ACT**: choose an object immediately
  - **ASK**: request clarification from the human

The key contribution is a **cost-sensitive policy** that makes this decision using expected utility.

---

## Why this is an HRI problem

This is a human-robot interaction problem because the robot is not just classifying language. It is deciding how to **interact with a human under uncertainty**.

The robot must balance three things:
- **task success**: choosing the correct object
- **interaction cost**: avoiding unnecessary questions
- **uncertainty**: recognizing when the instruction is ambiguous

This kind of reasoning is important in:
- assistive robotics
- collaborative robots in factories
- voice assistants
- embodied AI agents
- shared autonomy systems

---

## Problem formulation

The human's true intended object is hidden from or uncertain to the robot. The robot only observes the human's instruction, and possibly a later clarification response.

Because the true goal is not directly observable, this is modeled as a **Partially Observable Markov Decision Process (POMDP)**.

We define the problem as:

$$
P = (S, A, O, T, Z, R, \gamma)
$$

where:

- `S` = hidden states
- `A` = actions available to the robot
- `O` = observations available to the robot
- `T` = transition model
- `Z` = observation model
- `R` = reward function
- `gamma` = discount factor

In practice, I solve the problem in **belief space**, which gives a **belief-state MDP**.

---

## Hidden state

The hidden state is the human's intended object.

$$
s = g \in G
$$

where `G` is the set of candidate objects currently available in the environment.

For example:

$$
G = \{ \text{red mug}, \text{blue mug}, \text{bottle}, \text{notebook} \}
$$

The robot does not directly observe which object in `G` is the true goal.

---

## Action space

The robot can choose from the following actions:

$$
A = \{ ACT, ASK \}
$$

### ACT
The robot chooses an object immediately.

### ASK
The robot asks the human a clarification question before acting.

For example:
- "Did you mean the red mug or the blue mug?"

I intentionally keep the action space simple so that we can isolate the core decision problem:
**act now vs gather more information**

---

## Observations

The robot receives observations from the human.

$$
o \in O
$$

In this project, observations include:

- the original natural-language instruction
- a clarification response if the robot asks

Examples:
- instruction: "get me the mug"
- clarification response: "the blue one"

---

## Transition model

We assume that the human's underlying intent does not change during a short interaction.

So the transition model is:

$$
T(s' \mid s, a) = 1 \text{ if } s' = s
$$

This means the true goal remains fixed while the robot reasons about it.

---

## Observation model and language grounding

The robot needs a way to convert natural language into uncertainty over objects.

Ideally, this would be represented by an observation model:

$$
Z(o \mid s)
$$

which describes how likely an observation is given the true goal.

In the implementation, I approximate this step using a language model.

Given:
- a user instruction
- a list of candidate objects

the language model outputs a probability distribution over those objects.

I interpret that distribution as the robot's belief over possible goals.

---

## Belief state

Because the true state is hidden, the robot reasons using a belief distribution over candidate goals.

$$
b(g) = P(g \mid instruction)
$$

This means:

- `b(g)` is the probability that object `g` is the user's intended object
- the probabilities across all candidate objects sum to 1

For example, if the user types:

> get me the mug

the robot might form the following belief:

- red mug: 0.45
- blue mug: 0.45
- bottle: 0.05
- notebook: 0.05

This belief captures uncertainty explicitly.

The robot does not need to know the true object with certainty. It only needs a calibrated estimate of how likely each candidate is.

In the implementation, the policy layer validates this belief representation directly. It accepts any non-empty mapping from candidate objects to non-negative weights and normalizes those weights into a proper probability distribution before computing utilities. This keeps the decision logic mathematically consistent even if an upstream component provides unnormalized scores instead of already-calibrated probabilities.

---

## Why belief-state MDP is useful

Instead of reasoning over the hidden state directly, the robot reasons over the belief.

So the effective state becomes:

$$
s_t = b_t
$$

This converts the original POMDP into a **belief-state MDP**.

This is useful because the robot's decision can now be based directly on the current probability distribution over possible goals.

---

## Reward function

To decide whether to act or ask, the robot needs a reward model.

We define:

- `Rc` = reward for choosing the correct object
- `Cw` = cost of choosing the wrong object
- `Ca` = cost of asking a clarification question

These values encode the task tradeoff.

### Reward for ACT

If the robot acts and selects an object, it receives:
- a positive reward if the selected object matches the true intended goal
- a penalty if the selected object is incorrect

Let `g` denote the true intended object, and let `\hat{g}` denote the object selected by the robot.

Then:

$$
R(ACT, g, \hat{g}) =
\begin{cases}
Rc & \text{if } \hat{g} = g \\
-Cw & \text{if } \hat{g} \ne g
\end{cases}
$$

### Reward for ASK

If the robot asks a clarification question, it incurs a fixed interaction cost:

$$
R(ASK) = -Ca
$$

This reflects interruption, delay, or annoyance.

---

## Typical reward values

In the experiments, I use values such as:

- `Rc = 10`
- `Cw = 12`
- `Ca = 2`

This means:
- correctly acting is good
- acting incorrectly is somewhat worse than the benefit of being correct
- asking is mildly costly, but much cheaper than a wrong action

These values can also be varied in sensitivity analysis.

---

## Action-value functions

The robot chooses the action with the highest expected value.

Let:

$$
p^* = \max_g b(g)
$$

This is the probability of the most likely object under the current belief.

In other words:
- `p*` is the robot's current confidence in its best guess

---

## Expected utility of ACT

If the robot acts, it will choose the object with highest probability.

The expected utility of acting now is:

$$
Q(ACT) = p^* \cdot Rc - (1 - p^*) \cdot Cw
$$

### Intuition

This equation has two parts:

- `p* * Rc` = expected reward if the robot is correct
- `(1 - p*) * Cw` = expected penalty if the robot is wrong

So the robot acts only when its confidence is high enough to justify the risk.

### Example

Suppose:

- `p* = 0.45`
- `Rc = 10`
- `Cw = 12`

Then:

$$
Q(ACT) = 0.45 \cdot 10 - 0.55 \cdot 12 = 4.5 - 6.6 = -2.1
$$

So acting immediately is not attractive in this case.

---

## Expected utility of ASK

In the simplified one-step version of the project, asking has fixed immediate cost:

$$
Q(ASK) = -Ca
$$

If `Ca = 2`, then:

$$
Q(ASK) = -2
$$

This means asking is costly, but can still be better than making a risky action.

### More general interpretation

A more complete formulation would include the future benefit of clarification:

$$
Q(ASK) = -Ca + E[V(b')]
$$

where:
- `b'` is the updated belief after clarification
- `V(b')` is the value of acting from that improved belief state

In this project, we use the simpler one-step version to keep the decision problem tractable and interpretable.

---

## Policy definitions

We compare several policies.

### 1. Never Ask policy

The robot always acts immediately:

$$
\pi(b) = ACT
$$

This ignores uncertainty completely.

#### Strength
- fast
- no interruptions

#### Weakness
- prone to mistakes in ambiguous situations

---

### 2. Always Ask policy

The robot always asks before acting:

$$
\pi(b) = ASK
$$

This is highly cautious.

#### Strength
- avoids many wrong actions

#### Weakness
- can be slow and annoying

---

### 3. Fixed Threshold policy

The robot acts only if its confidence exceeds a manually chosen threshold `tau`:

$$
\pi(b) =
\begin{cases}
ACT & \text{if } p^* \ge \tau \\
ASK & \text{otherwise}
\end{cases}
$$

For example, if `tau = 0.75`, then the robot only acts when its top belief is at least 75%.

#### Strength
- simple and intuitive

#### Weakness
- the threshold is arbitrary
- it does not adapt to the cost structure of the task

---

### 4. Cost-sensitive policy (main method)

The main method compares expected utility directly:

$$
\pi(b) = \arg\max_{a \in \{ACT, ASK\}} Q(a)
$$

That means:
- compute the expected value of acting
- compute the expected value of asking
- choose the action with the higher value

This is the main contribution of the project.

#### Why it is better
Unlike a fixed threshold, this policy is grounded in the actual reward structure:
- if wrong actions are expensive, it asks more
- if asking is expensive, it asks less
- if confidence is high, it acts

---

## Derived threshold from costs

An important insight is that the cost-sensitive policy implies a closed-form confidence threshold.

We start from the condition:

$$
Q(ACT) \ge Q(ASK)
$$

Substituting the value functions:

$$
p^* \cdot Rc - (1 - p^*) \cdot Cw \ge -Ca
$$

Solving for `p*` gives:

$$
p^* \ge \frac{Cw - Ca}{Rc + Cw}
$$

This is the **derived threshold**.

### Why this matters

This shows that the decision boundary is not arbitrary. It emerges from the task costs.

For example, if:

- `Rc = 10`
- `Cw = 12`
- `Ca = 2`

then:

$$
p^* \ge \frac{12 - 2}{10 + 12} = \frac{10}{22} \approx 0.455
$$

So the robot should act whenever confidence exceeds about 45.5%.

This is more principled than manually choosing a threshold such as 0.75.

---

## Relationship between cost-sensitive and derived-threshold policies

These two are closely related.

- The **cost-sensitive policy** computes expected utility directly
- The **derived-threshold policy** is the algebraic simplification of that comparison under the one-step reward model

So in the current project setup, they are mathematically equivalent.

However, the cost-sensitive formulation is more general and more extensible. If we later add:
- future belief updates
- richer clarification value
- additional actions

then the cost-sensitive formulation still works, while the threshold shortcut may no longer apply.

---

## Human model

So far, the clarification cost `Ca` has been treated as a fixed constant. In practice, the cost of asking a human depends on who the human is and how many times they have already been interrupted.

This project extends the framework with a **human model** that makes `Ca` dynamic.

---

### Motivation

A fixed `Ca` assumes every clarification request has the same cost to the human. But consider:

- A **patient** user may not mind repeated questions.
- A **busy** user finds even the first question disruptive.
- An **interruption-averse** user becomes increasingly annoyed with each successive ask.

To capture this, we introduce a **fatigue model** in which the effective clarification cost grows over the course of a session.

---

### Dynamic clarification cost

The effective clarification cost at step `t` is defined as:

$$
Ca(t) = Ca_{\text{base}} + \text{fatigue\_rate} \times n_{\text{asks}}
$$

where:

- `Ca_base` is the baseline cost of a single clarification request for this human
- `fatigue_rate` is the additional cost added per prior interruption in the session
- `n_asks` is the number of ASK actions the robot has already taken in the current session

This means the robot pays a higher cost for each successive question it asks, which naturally discourages over-asking as the session progresses.

---

### Human profiles

Three profiles are provided, each encoding a different tolerance for clarification:

| Profile | `Ca_base` | `fatigue_rate` | Description |
|---|---|---|---|
| `patient` | 1.0 | 0.1 | Tolerant of questions; fatigue builds very slowly |
| `busy` | 5.0 | 0.5 | High baseline cost; moderate fatigue growth |
| `interruption_averse` | 3.0 | 1.5 | Moderate baseline, but cost grows rapidly with repeated asks |

For example, with the `interruption_averse` profile after two prior asks:

$$
Ca(2) = 3.0 + 1.5 \times 2 = 6.0
$$

---

### Effect on the decision policy

The dynamic `Ca(t)` feeds directly into both the cost-sensitive and derived-threshold policies.

**Expected utility of ASK** becomes session-dependent:

$$
Q(ASK, t) = -Ca(t)
$$

**Derived threshold** also changes with `t`:

$$
p^*(t) \ge \frac{Cw - Ca(t)}{Rc + Cw}
$$

As `Ca(t)` grows, `Q(ASK)` becomes more negative and the threshold for acting decreases. This means a robot that has already asked several times will prefer to act rather than ask again, even under moderate uncertainty. The policy adapts to the human's state without requiring any explicit human feedback beyond the profile selection.

---

### Worked example

Suppose `Rc = 10`, `Cw = 12`, and the human profile is `interruption_averse` with `Ca_base = 3.0`, `fatigue_rate = 1.5`.

**First ask (n_asks = 0):**

$$
Ca(0) = 3.0 + 1.5 \times 0 = 3.0
$$
$$
\text{Derived threshold} = \frac{12 - 3.0}{10 + 12} = \frac{9}{22} \approx 0.409
$$

**After two prior asks (n_asks = 2):**

$$
Ca(2) = 3.0 + 1.5 \times 2 = 6.0
$$
$$
\text{Derived threshold} = \frac{12 - 6.0}{10 + 12} = \frac{6}{22} \approx 0.273
$$

With a lower threshold, the robot becomes more willing to act even under uncertainty, because asking a third time is now very costly.

---

## Experimental hypothesis

We test the following central hypothesis:

> A cost-sensitive policy that reasons explicitly about uncertainty and interaction cost will outperform heuristic baselines in balancing correctness and efficiency.

More specifically:

### Hypothesis 1
The cost-sensitive policy will achieve higher total utility than:
- Never Ask
- Always Ask
- Fixed Threshold

### Hypothesis 2
The cost-sensitive policy will reduce unnecessary clarifications compared to Always Ask.

### Hypothesis 3
The cost-sensitive policy will reduce incorrect actions compared to Never Ask.

### Hypothesis 4
The derived-threshold interpretation will provide an interpretable explanation of robot behavior in terms of task costs.

---

## Experimental setup

Each trial contains:
- a set of candidate objects
- a human instruction
- a language-model-generated belief distribution over those objects
- a ground-truth intended object for evaluation

The robot receives the belief and applies one of the policies.

If it chooses:
- **ACT**: it selects the highest-probability object
- **ASK**: it requests clarification and incurs asking cost

---

## Evaluation metrics

We evaluate policies using the following metrics.

### 1. Task success rate
Fraction of trials in which the robot ultimately chooses the correct object.

### 2. Number of clarifications
How often the robot asks the user for more information.

### 3. Total utility
Sum of rewards over trials, combining:
- correct decisions
- wrong actions
- clarification costs

### 4. Error rate
Fraction of action trials in which the robot selects the wrong object.

### 5. Entropy of belief
We also track uncertainty in the belief distribution:

$$
H(b) = - \sum_g b(g) \log b(g)
$$

Higher entropy means greater ambiguity.

This is useful for analyzing whether the robot asks more often in genuinely uncertain situations.

---

## Why the LLM matters

A major part of the project is that I do not hand-code the mapping from language to object probabilities.

Instead, I use a language model to estimate:

$$
b(g) \approx P(g \mid instruction, G)
$$

This makes the system much more flexible:
- objects can be added or removed dynamically
- instructions can vary naturally
- ambiguity is handled probabilistically rather than with hard-coded rules

This is important because modern HRI systems increasingly rely on language models for semantic grounding.

---

## Key contribution

The project combines four elements:

1. **LLM-based probabilistic language grounding**  
   The robot uses a language model to convert human instructions into a belief distribution over objects.

2. **Belief-state decision-making**  
   The robot reasons in belief space rather than assuming a known goal.

3. **Cost-sensitive clarification policy**  
   The robot decides whether to act or ask by maximizing expected utility.

4. **Human model with dynamic clarification cost**  
   A fatigue model encodes how the cost of asking grows with repeated interruptions, parameterized by human profiles (patient, busy, interruption-averse). The policy adapts its ask threshold in real time as the session progresses.

Together, these provide a simple but principled framework for interaction under uncertainty that is sensitive to both task costs and human state.

---

## Summary

This project asks a simple but important question:

**When should a robot act, and when should it ask?**

I model the problem as a POMDP because the human's intended goal is hidden. I solve it as a belief-state MDP by maintaining a probability distribution over candidate objects. A language model provides the belief distribution from natural-language instructions. The robot then uses a cost-sensitive policy to compare the expected value of acting versus asking for clarification.

The framework is extended with a human model that makes the clarification cost dynamic. Each human profile encodes a baseline interruption cost and a fatigue rate, so the effective cost of asking grows with the number of prior questions in a session. This causes the robot to ask less aggressively over time when interacting with an impatient or interruption-averse user, without requiring explicit feedback from the human.

This framework could be useful because it makes ambiguity explicit, grounds decisions in uncertainty, connects robot behavior directly to task costs, and adapts to individual human tolerance for clarification.
