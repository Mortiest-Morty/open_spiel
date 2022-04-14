# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy gradient agents trained and evaluated on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient_ppo

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e6), "Number of train episodes.")
flags.DEFINE_integer("eval_every", int(1e4), "Eval agents every x episodes.")
flags.DEFINE_bool("save_checkpoints", False, "Save neural network weights or not.")
flags.DEFINE_integer("save_every", int(5e4), "Save checkpoint of agents' models every x episodes.")
flags.DEFINE_enum("loss_str", "ppo", ["a2c", "rpg", "qpg", "rm", "ppo"],
                  "PG loss to use.")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("batch_size", 16, "batchsize of network")
flags.DEFINE_integer("seed_", 10, "set seed for random")
flags.DEFINE_string("checkpoint_dir", "checkpoints/",
                    "Directory to save/load the agent.")
flags.DEFINE_bool("load_checkpoints", False, "load neural network weights.")
flags.DEFINE_integer("load_idx", -1, "the exact index of the neural network weights to load")
flags.DEFINE_integer("num_players", 2, "number of players in the setup game")
flags.DEFINE_float("critic_lr", 0.001, "learning rate of critic network. Also learning rate of ACH.")
flags.DEFINE_float("pi_lr", 0.001, "learning rate of policy network")
flags.DEFINE_float('etpcost', 0.01, "entropy cost")
flags.DEFINE_integer('nctopi', 1, "num critic before pi")
flags.DEFINE_string("optm", "sgd", "optimizer for training")
flags.DEFINE_float("clip_param", 0.2, "threshold for clip method")
flags.DEFINE_float("gae_lamda", 0.95, "lamda parameter for gae advantage")


class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(_):
  
  gpu_list = [0]
  os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(
        str(gpu) + ',' for gpu in gpu_list)
  
  game = FLAGS.game_name
  num_players = FLAGS.num_players

  np.random.seed(FLAGS.seed_)
  tf.set_random_seed(FLAGS.seed_)
  
  env_configs = {"players": num_players}
  chance_event_sampler = rl_environment.ChanceEventSampler(seed=FLAGS.seed_)
  env = rl_environment.Environment(game, **env_configs, chance_event_sampler=chance_event_sampler)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        policy_gradient_ppo.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=FLAGS.loss_str,
            batch_size=FLAGS.batch_size,
            hidden_layers_sizes=(128, ),
            critic_learning_rate=FLAGS.critic_lr,
            pi_learning_rate=FLAGS.pi_lr,
            entropy_cost=FLAGS.etpcost,
            clip_param=FLAGS.clip_param,
            num_critic_before_pi=FLAGS.nctopi,
            optimizer_str=FLAGS.optm,
            gae_lamda=FLAGS.gae_lamda) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)

    
    if FLAGS.load_checkpoints:
      load_dir = FLAGS.checkpoint_dir + "iter_" + str(FLAGS.load_index)
      for index, agent in enumerate(agents):
        if agent.has_checkpoint(load_dir):
          agent.restore(load_dir)
          print("load weights: iter_" + str(FLAGS.load_index) + "_player_" + str(index) + " success!")
        else:
          print("fail to load neural network weights: iter_" + str(FLAGS.load_index))
          break
    
    
    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_episodes):

      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
        logging.info("%s", msg)
      
      if FLAGS.save_checkpoints and (ep + 1) % FLAGS.save_every == 0:
        save_dir = FLAGS.checkpoint_dir + "iter_" + str(ep + 1)
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)
        for agent in agents:
          agent.save(save_dir)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)