---
layout: distill
title: LLMs in JAX from scratch - Part1
# description: an example of a distill-style blog post and main elements
tags: llm
giscus_comments: true
date: 2026-11-04
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Amine Andam
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: UM6P

bibliography: 2026-11-04-tinyjxllm-p1

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Interactive Plots
  - name: Mermaid
  - name: Diff2Html
  - name: Leaflet
  - name: Chartjs, Echarts and Vega-Lite
  - name: TikZ
  - name: Typograms
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

In this series of blog posts, we will build and train LLMs from scratch using JAX. By “from scratch,” I mean using only pure JAX (JAX arrays, pytrees, etc.), without using any other framework (e.g., Flax). We will create a tiny neural network library that implements all the components needed to train LLMs.

I designed these blog posts in a way that reflects how I like to learn things. For me, *to learn something* means that I have acquired enough detail to:

1. Build internal mental models of the topic. This means being able to mentally visualize the things I have learned. For example, I can visualize the mathematical equations of a transformer layer in my head.
2. Reason about the topic from memory, without looking at my notes or putting a lot of effort into recalling the information.

To achieve this for math and CS-related topics, I need the following:

1. *A detailed mathematical description of the topic:* I like content (especially books and long lecture notes) that explains every detail and provides all the necessary background and proofs to understand the new concepts. More often than not, missing prerequisites are the main reason I don't understand something ( for more read this [Advice on Upskilling](https://docs.google.com/document/d/17qFY5w9uEWL4VVyJSTEmSSRyvegh_Y4N6NMSxwCmGaE/edit?tab=t.0#heading=h.cnsdcie0zjue)). Therefore, I avoid investing my time in lecture slides or illustrated guides, as they don't provide the level of detail I need. Moreover, I prefer mathematical explanations that come with worked examples. My best strategy for fully grasping the math I want to learn is to repeatedly read the explanations and worked examples, and manually derive all the equations using pen and paper.

2. *Implement it yourself:* Some topics may look intimidating and very complex at first glance. A technique that helps me get past this psychological barrier is implementing things from scratch. For example, when I first started learning Deep Reinforcement Learning (DRL), I thought it must be very difficult to implement algorithms developed by researchers at top AI labs (e.g., DeepMind, OpenAI) by myself. There were also so many algorithms. But this turned out not to be the case: every RL algorithm is, roughly, just a way to collect samples + an optimization loop. It is enough to understand two algorithms (DQN and PPO), and the other algorithms use the same techniques (replay buffers, TD errors, etc.). A second example is multi-agent RL: you can understand most MARL papers just by knowing what the input to each neural network is. If you are interested in clean implementations of RL/MARL algorithms, you can check out [CleanMARL](https://github.com/AmineAndam04/cleanmarl), [MARL docs](https://cleanmarl-docs.readthedocs.io/en/latest/marl.htm), and [FlaxRL](https://github.com/AmineAndam04/flaxrl).

3. *Understand the shapes:* A sign that you understand your implementation is being able to infer the shapes of your tensors just by reading the code, especially after not seeing it for a while. Moreover, you should be able to remember why you applied transformations such as flattening, transposing, squeezing, and unsqueezing.
