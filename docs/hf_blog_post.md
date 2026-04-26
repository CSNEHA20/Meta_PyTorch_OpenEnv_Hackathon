---
title: "How We Taught AI to Dispatch Ambulances: A Journey into Life-Saving Machine Learning"
thumbnail: https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV/resolve/main/agent_comparison.png
authors:
  - user: CSNEHA20
  - user: Vishallakshmikanthan
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - trl
  - ambulance
  - multi-agent
  - emergency-services
---

# How We Taught AI to Dispatch Ambulances: A Journey into Life-Saving Machine Learning

![Ambulance Dispatch Simulation](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV/resolve/main/agent_comparison.png)

> Every minute counts in an emergency. When someone's life hangs in the balance, getting an ambulance to them quickly can mean the difference between life and death. Studies show that every 60-second delay in critical cases increases mortality by 10%. This stark reality is what drove us to ask: Can artificial intelligence help save lives by making better dispatch decisions?

## The Problem We Set Out to Solve

Picture this: You're an emergency dispatcher in India, handling one of over 40 million emergency calls that come through the 108 and 112 networks every year. On your screen, multiple emergencies are flashing red. You have ambulances scattered across the city, hospitals with varying capacities, and traffic conditions that change by the minute. 

Who do you send first? Which ambulance is actually closest right now? Which hospital can handle this specific type of emergency? These aren't just logistical puzzles — they're life-or-death decisions made under immense pressure.

We realized that this is exactly the kind of complex, time-sensitive decision-making challenge that modern AI could potentially help with. But first, we needed to create a way to teach AI how to think like an experienced dispatcher.

## Building a Virtual City to Train AI

Our first challenge was creating a realistic training ground. We couldn't exactly let untrained AI experiment on real emergency calls! So we built **Ambulance-OpenENV** — a sophisticated simulation that mirrors the chaos and complexity of real-world ambulance dispatch.

### What Goes Into Our Virtual City?

**A Realistic Road Network:** We modeled our city after actual urban layouts — not just a simple grid, but a complex network with busy hubs and quieter connecting roads, just like real cities have downtown cores and residential neighborhoods.

**A Fleet with Personality:** Each ambulance in our simulation isn't just a dot on a map. Every vehicle has its own state — it could be waiting at the station, rushing to an emergency, at the scene providing care, transporting a patient, returning from a hospital, or even strategically repositioning to be ready for the next call.

**Emergencies Aren't All Equal:** We built in three levels of urgency — critical emergencies where every second counts (like cardiac arrests), high-priority cases that need quick attention, and normal emergencies. The system constantly generates these calls, creating that overwhelming feeling dispatchers know all too well.

**The Real World is Messy:** We added rush hour traffic that slows everything down, random road incidents that block routes, and even specialized hospitals — because sending a trauma patient to a general hospital when a trauma center is available isn't just inefficient, it can cost lives.

## Teaching AI to Make Moral Trade-offs

Here's where things get interesting. How do you teach an AI to prioritize? If you have two emergencies and only one ambulance, which one do you send it to?

We created what we call a "Rubric" — essentially a scoring system that guides the AI toward good decisions:

- **Saving lives gets the biggest rewards.** Successfully reaching an emergency scores points, with bonuses for handling critical cases quickly.
- **Efficiency matters too.** Getting patients to the right hospital, dispatching quickly, and keeping the fleet moving all earn points.
- **Penalties teach consequences.** Points are deducted when emergencies go unserved, ambulances sit idle while people are waiting, or patients get sent to hospitals that can't help them.

This isn't just about getting a high score — it's about teaching the AI to internalize the values we want it to have: prioritize critical cases, never leave people waiting unnecessarily, and always route intelligently.

## Two Approaches to Training Our AI Dispatchers

We experimented with two different ways of teaching our AI to dispatch:

### The Deep Learning Approach: DQN

Think of this like training a dispatcher through thousands of simulated shifts. Our DQN agent learns by doing — trying different strategies, seeing what works, and gradually building up intuition about good decisions. It processes 124 pieces of information every moment: where every ambulance is, how urgent each emergency is, which hospitals have room, and what the traffic looks like.

After hundreds of simulated episodes, it starts developing instincts — knowing, for example, that it's often worth sending an ambulance from slightly further away if the closer one is about to finish a transport and will be available sooner.

### The Language Model Approach: GRPO

This is where it gets really exciting. We wondered: Could we take a language model — the same technology behind chatbots — and teach it to dispatch ambulances?

Language models process information as text, so we describe the entire situation to them in structured messages: "Ambulance 3 is at coordinates X,Y with 2 passengers, Emergency 7 is CRITICAL at location Z..." The model has to output a valid dispatch decision in response.

We used a cutting-edge training technique called GRPO (Group Relative Policy Optimization) through HuggingFace's TRL library. This lets the model try different approaches, compare them against each other, and learn which strategies work best. We even used Unsloth for efficient training, making it possible to run on free Google Colab GPUs.

The model we trained — Qwen2.5-0.5B-Instruct — is relatively small (just half a billion parameters), making it fast and deployable even on modest hardware.

## What We Learned: Can AI Actually Dispatch?

So, did it work? Let's look at the numbers.

**Random Dispatching** (essentially doing nothing smart): Almost zero success rate. Emergencies go unserved, ambulances wander aimlessly.

**Simple Greedy Rules** (always send the closest ambulance): About 40% success on easy scenarios, but falls apart when things get complex. This is like a new dispatcher who knows the basics but lacks experience.

**Our Trained DQN Agent**: Climbed to around 60% success on easy scenarios and significantly outperformed the greedy approach on harder tasks. It's learning patterns that simple rules miss.

**Our GRPO-Trained Language Model**: Showed clear improvement over the baseline. The reward curves trend upward, and in head-to-head comparisons, it consistently beats the simple greedy approach.

![Training Progress](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV/resolve/main/reward_curve.png)

*Our DQN agent steadily improved over 500 training episodes, learning to make better dispatch decisions.*

![GRPO Results](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV/resolve/main/grpo_before_after.png)

*The language model started with random behavior but progressively learned valid, effective dispatch strategies.*

## What Did the AI Actually Learn?

Watching our trained agents work is fascinating. They developed several sophisticated behaviors that weren't explicitly programmed:

**Smart Triage:** The AI learned to prioritize critical emergencies over less urgent ones, even when it meant making a slightly longer drive. It figured out that a 2-minute delay to a critical case is much worse than a 5-minute delay to a normal case.

**Hospital Matching:** It figured out that trauma centers are for critical injuries, cardiac units for heart problems. The AI routes patients to appropriate facilities, not just the nearest one.

**Fleet Awareness:** The AI keeps track of which ambulances are truly available and which just look free on paper. It learned to anticipate when an ambulance will finish its current job and be ready for the next call.

**Proactive Positioning:** Perhaps most impressively, the AI learned to move idle ambulances to predicted hotspots between calls — positioning resources before emergencies happen.

These aren't programmed rules. The AI developed these strategies through trial and error, discovering insights that took human dispatchers years of experience to develop.

## Why This Matters Beyond the Simulation

Our project demonstrates something important: AI can learn to handle complex, multi-faceted decision-making in high-stakes environments. The techniques we used — combining reinforcement learning with language models — could apply to other resource allocation problems.

But we want to be clear-eyed about limitations. Our simulation, while sophisticated, is still a simplification of the real world. Real dispatch involves human judgment, unpredictable factors, and ethical considerations that go beyond what we can simulate. We see this as a proof-of-concept — showing what's possible, not a ready-to-deploy system.

## Try It Yourself

Curious to see our AI dispatchers in action? You can:

**Run it locally:** Clone our repository, install the requirements, and watch the AI make real-time dispatch decisions.

```bash
git clone https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon.git
cd Meta_PyTorch_OpenEnv_Hackathon
pip install -r requirements.txt
python inference.py
```

**Try the live demo:** Visit our [HuggingFace Space](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) to see real-time dispatch visualization.

**Train your own:** Open our [Colab notebook](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb) and train a GRPO model yourself — it runs in about 30 minutes on free Google Colab hardware.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb)

## The Road Ahead

We're excited about where this could go. Future work could include:

- More realistic city models with actual traffic data
- Integration with real emergency call patterns
- Multi-agent coordination between dispatch centers
- Better handling of edge cases and rare emergencies

## Final Thoughts

Building Ambulance-OpenENV was more than a technical exercise. It was a journey into how AI can learn to make decisions that genuinely matter. Every improvement in our agents' performance represents potential lives saved in the real world.

We believe that thoughtfully designed AI, trained with the right values and carefully validated, could one day support human dispatchers in making better decisions faster. Not replacing human judgment, but augmenting it — giving experienced professionals AI assistants that have "seen" millions of scenarios and can offer suggestions grounded in pattern recognition no human could develop alone.

That's a future worth building toward.

---

*This project was built for the Scaler × Meta × HuggingFace × PyTorch OpenEnv Hackathon 2026*

*Team: CSNEHA20 and Vishallakshmikanthan*

**Resources:**
- 🤗 [HuggingFace Space Demo](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV)
- 🐙 [GitHub Repository](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon)
- 📓 [Full Training Notebook](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb)
