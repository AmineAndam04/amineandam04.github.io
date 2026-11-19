---
layout: distill
title: How I like to learn math and computer science
description: The explanations, implementations, and tensor shapes that help me understand a topic.
date: 2026-11-18
tags: learning math computer-science
giscus_comments: true
authors:
  - name: Amine Andam
    affiliations:
      name: UM6P
---

For me, _learning something_ means understanding it well enough to:

1. Build a mental model of the topic. For example, I should be able to visualize the computations in a transformer layer.
2. Reason about the topic from memory, without consulting my notes or struggling to recall the details.

For mathematics and computer science topics, three things help me reach that level of understanding:

1. _Detailed mathematical explanations:_ I like content (especially books and long lecture notes) that explains every detail and provides all the necessary background and proofs to understand the new concepts. More often than not, missing prerequisites are the main reason I don't understand something (for more on this, see [Advice on Upskilling](https://docs.google.com/document/d/17qFY5w9uEWL4VVyJSTEmSSRyvegh_Y4N6NMSxwCmGaE/edit?tab=t.0#heading=h.cnsdcie0zjue)). Therefore, I avoid investing my time in lecture slides or illustrated guides, as they don't provide the level of detail I need. Moreover, I prefer mathematical explanations that come with worked examples. My best strategy for fully grasping the math I want to learn is to reread the explanations and worked examples, then derive the equations with pen and paper.

2. _Implementing the ideas from scratch:_ Some topics may look intimidating and very complex at first glance. A technique that helps me get past this psychological barrier is implementing things from scratch. For example, when I first started learning deep reinforcement learning (DRL), I thought it must be very difficult to implement algorithms developed by researchers at top AI labs (e.g., DeepMind, OpenAI) by myself. There were also so many algorithms. Implementing them made the task feel more manageable: many RL algorithms combine a way to collect samples with an optimization loop. Understanding DQN and PPO gave me a foundation for recognizing shared techniques, such as replay buffers and temporal-difference errors. In multi-agent RL (MARL), identifying the inputs to each neural network also helps me understand how an algorithm works. If you are interested in clean implementations of RL/MARL algorithms, you can check out [CleanMARL](https://github.com/AmineAndam04/cleanmarl), [MARL docs](https://cleanmarl-docs.readthedocs.io/en/latest/marl.htm), and [FlaxRL](https://github.com/AmineAndam04/flaxrl).

3. _Understanding tensor shapes:_ I consider an implementation familiar when I can infer tensor shapes just by reading the code, even after some time away. I should also be able to explain why I used transformations such as flattening, transposing, squeezing, and unsqueezing.
