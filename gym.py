from agent import SmartAgent
import models
import game
import pandas as pd

model = models.PolicyNet()

agent_1 = SmartAgent(model)
agent_2 = SmartAgent(model)

gamma = .9

def get_q_table(agent:SmartAgent, result):
    states = agent.states.copy()
    states.reverse()

    actions = agent.actions.copy()
    actions.reverse()

    data = []

    for i in range(len(actions)):
        data.append((states[i], actions[i], result))
        result = result * gamma
    return pd.DataFrame(columns=['state', 'action', 'result'], data=data)



for _ in range(10):
    g = game.Game(agent_1, agent_2)
    result = g.start()
    result_origin = result

    data = []

    q = pd.concat(
        [
            get_q_table(agent_1, result),
            get_q_table(agent_2, -result)
        ],
        ignore_index=True
    )

    model.train(q)

# https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24

    