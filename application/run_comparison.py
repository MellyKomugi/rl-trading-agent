


#telecharger les données de train_env
#2. Entrainer les agents 
#3; COnstruire les données de test_env
#4. Lancer les predictions 
#5. Recuperer les resultats 
#6. Comparer les performances
#7. sauvegarder les graphiques et CSV



from environment.setup_env import build_envs

# from agents.reinforce import ReinforceAgent
# from agents.exp3 import EXP3Agent
from agents.DRL import DRLAgent
from agents.architectures import SimplePortfolioMLP


def run_comparison(
    if_using_exp3 : bool=True , 
    if_using_reinforce : bool =True, 
    if_using_dqn : bool =True,
    if_using_policy_gradient : bool =True,
    if_using_ppo : bool =False
) : 
    train_env , test_env, train_df , test_df =build_envs()


    # type of env
    print(type(train_env))

    # if if_using_dqn : 
    #     continue
    # if if_using_exp3 : 
    #     continue  
    if if_using_ppo : 
        agent=DRLAgent(env=train_env)
        PPO_PARAMS={
            "n_steps" : 20, 
            "ent_coef": 0.01, 
            "learning_rate" : 0.00025 , 
            "batch_size": 128,
        }
        model_ppo =agent.get_model("ppo", model_kwargs=PPO_PARAMS)

        trained_ppo=agent.train_model(model_ppo, total_timesteps=5000)

    if if_using_policy_gradient:
        agent = DRLAgent(env=train_env)

        pg_model = agent.get_model(
            "pg",
            policy=SimplePortfolioMLP,
            device="cpu",
            model_kwargs={
                "batch_size": 64,
                "lr": 1e-3,
                "action_noise": 0.01,
            },
            policy_kwargs={
                "input_shape": (7, 3),
                "portfolio_size": 3,
                "hidden_dim": 64,
                "device": "cpu",
            },
        )
        trained_pg = agent.train_model(pg_model, episodes=50)

        results_pg, actions_pg = DRLAgent.DRL_prediction(
            model=trained_pg,
            environment=test_env,
            online_training_period=10**9,
            lr=0,
        )

        actions_pg.to_csv("actions_pg.csv", index=False)
        results_pg.to_csv("results_pg.csv", index=False)

    return results_pg, actions_pg


    # #trade with trained agents 

    # if if_using_ppo: 
    #     results_ppo, actions_ppo=DRLAgent.DRL_prediction(
    #         model=trained_ppo, environment=test_env
    #     )


    # if if_using_ppo and isinstance(results_ppo, tuple) : 
    #     actions_ppo=results_ppo[1]
    #     results_ppo=results_ppo[0]


    # #store actions 
    # actions_ppo.to_csv('actions_ppo.csv') if if_using_ppo else None





