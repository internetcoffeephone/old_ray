"""Note: Keep in sync with changes to VTracePolicyGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gym

import ray
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override


def agent_name_to_idx(name):
    agent_num = int(name[6])
    return agent_num


class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff -
                           self.entropy * entropy_coeff)


class CuriosityLoss(object):
    def __init__(self, pred_states, true_states, loss_weight=1.0):
        """Curiosity loss with supervised MSE loss on a trajectory.

        The loss is based on the difference between the predicted encoding of the observation x at t+1 based on t,
         and the true encoding x at t+1.
         The loss is then -log(p(xt+1)|xt, at)
         Difference is measured as mean-squared error corresponding to a fixed-variance Gaussian density.

        Returns:
            A scalar loss tensor.
        """
        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        pred_states = pred_states[:-1, :]  # [B, N]

        # Remove first true state, as we have nothing to predict this from.
        # the t+1 actions of other agents from all actions at t.
        true_states = true_states[1:, :]

        # Compute mean squared error of difference between prediction and truth
        mse = tf.losses.mean_squared_error(true_states, pred_states)

        self.total_loss = mse * loss_weight
        tf.print("Curiosity loss", self.total_loss, [self.total_loss])


class A3CPolicyGraph(LearningRateSchedule, TFPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG, **config)
        self.config = config
        self.sess = tf.get_default_session()

        # Read curiosity options from config
        cust_opts = config['model']['custom_options']
        self.aux_loss_weight = cust_opts['aux_loss_weight']
        self.aux_reward_clip = cust_opts['aux_reward_clip']
        self.aux_reward_weight = cust_opts['aux_reward_weight']
        self.aux_curriculum_steps = cust_opts['aux_curriculum_steps']
        self.aux_scale_start = cust_opts['aux_scaledown_start']
        self.aux_scale_end = cust_opts['aux_scaledown_end']
        self.aux_scale_final_val = cust_opts['aux_scaledown_final_val']

        # Use to compute aux curriculum weight
        self.steps_processed = 0

        # Setup the policy
        self.observations = tf.placeholder(tf.float32,
                                           [None] + list(observation_space.shape))

        dist_class, self.num_actions = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        prev_actions = ModelCatalog.get_action_placeholder(action_space)
        prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")

        # Compute output size of curiosity model
        self.aux_dim =

        # We now create two models, one for the policy, and one for the model
        # of other agents (MOA)
        self.rl_model, self.curiosity_model = ModelCatalog.get_double_lstm_model({
                "obs": self.observations,
                "prev_actions": prev_actions,
                "prev_rewards": prev_rewards,
                "is_training": self._get_is_training_placeholder(),
            }, observation_space, self.num_actions, self.aux_dim,
            self.config["model"], lstm1_name="policy", lstm2_name="curiosity")

        action_dist = dist_class(self.rl_model.outputs)
        self.action_probs = tf.nn.softmax(self.rl_model.outputs)
        self.vf = self.rl_model.value_function()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

        # Setup the policy loss
        if isinstance(action_space, gym.spaces.Box):
            ac_size = action_space.shape[0]
            actions = tf.placeholder(tf.float32, [None, ac_size], name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            actions = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise UnsupportedSpaceException("Action space {} is not supported for A3C.".format(action_space))
        advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.v_target = tf.placeholder(tf.float32, [None], name="v_target")
        self.rl_loss = A3CLoss(action_dist, actions, advantages,
                               self.v_target,
                               self.vf,
                               self.config["vf_loss_coeff"],
                               self.config["entropy_coeff"])

        # Setup the MOA loss
        self.aux_preds = tf.reshape(  # Reshape to [B,N,A]
            self.moa.outputs, [-1, self.num_other_agents, self.num_actions])
        self.curiosity_loss = CuriosityLoss(self.aux_preds, self.others_actions,
                                            loss_weight=self.aux_loss_weight)
        self.moa_action_probs = tf.nn.softmax(self.moa_preds)

        # Total loss
        self.total_loss = self.rl_loss.total_loss + self.moa_loss.total_loss

        # Initialize TFPolicyGraph
        loss_in = [
            ("obs", self.observations),
            ("others_actions", self.others_actions),
            ("actions", actions),
            ("prev_actions", prev_actions),
            ("prev_rewards", prev_rewards),
            ("advantages", advantages),
            ("value_targets", self.v_target),
        ]
        if self.train_moa_only_when_visible:
            loss_in.append(('others_visibility', self.others_visibility))
        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.observations,
            action_sampler=action_dist.sample(),
            action_prob=action_dist.sampled_action_prob(),
            loss=self.total_loss,
            model=self.rl_model,
            loss_inputs=loss_in,
            state_inputs=self.rl_model.state_in + self.moa.state_in,
            state_outputs=self.rl_model.state_out + self.moa.state_out,
            prev_action_input=prev_actions,
            prev_reward_input=prev_rewards,
            seq_lens=self.rl_model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.total_influence = tf.get_variable("total_influence", initializer=tf.constant(0.0))

        self.stats = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "policy_loss": self.rl_loss.pi_loss,
            "policy_entropy": self.rl_loss.entropy,
            "grad_gnorm": tf.global_norm(self._grads),
            "var_gnorm": tf.global_norm(self.var_list),
            "vf_loss": self.rl_loss.vf_loss,
            "vf_explained_var": explained_variance(self.v_target, self.vf),
            "moa_loss": self.moa_loss.total_loss,
            "total_influence": self.total_influence
        }

        self.sess.run(tf.global_variables_initializer())

    @override(TFPolicyGraph)
    def copy(self, existing_inputs):
        # Optional, implement to work with the multi-GPU optimizer.
        raise NotImplementedError

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.rl_model.state_init + self.curiosity_model.state_init

    @override(TFPolicyGraph)
    def _build_compute_actions(self,
                               builder,
                               obs_batch,
                               state_batches=None,
                               prev_action_batch=None,
                               prev_reward_batch=None,
                               episodes=None):
        state_batches = state_batches or []
        if len(self._state_inputs) != len(state_batches):
            raise ValueError(
                "Must pass in RNN state batches for placeholders {}, got {}".
                format(self._state_inputs, state_batches))
        builder.add_feed_dict(self.extra_compute_action_feed_dict())

        # Extract matrix of other agents' past actions, including agent's own
        if type(episodes) == dict and 'all_agents_actions' in episodes.keys():
            # Call from visualizer_rllib, change episodes format so it complies with the default format.
            self_index = agent_name_to_idx(self.agent_id)
            # First get own action
            all_actions = [episodes['all_agents_actions'][self_index]]
            others_actions = [e for i, e in enumerate(
                episodes['all_agents_actions']) if self_index != i]
            all_actions.extend(others_actions)
            all_actions = np.reshape(np.array(all_actions), [1, -1])
        else:
            own_actions = np.atleast_2d(np.array(
                [e.prev_action for e in episodes[self.agent_id]]))
            all_actions = self.extract_last_actions_from_episodes(
                episodes, own_actions=own_actions)

        builder.add_feed_dict({self._obs_input: obs_batch,
                               self.others_actions: all_actions})

        if state_batches:
            seq_lens = np.ones(len(obs_batch))
            builder.add_feed_dict({self._seq_lens: seq_lens,
                                   self.moa.seq_lens: seq_lens})
        if self._prev_action_input is not None and prev_action_batch:
            builder.add_feed_dict({self._prev_action_input: prev_action_batch})
        if self._prev_reward_input is not None and prev_reward_batch:
            builder.add_feed_dict({self._prev_reward_input: prev_reward_batch})
        builder.add_feed_dict({self._is_training: False})
        builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
        fetches = builder.add_fetches([self._sampler] + self._state_outputs +
                                      [self.extra_compute_action_fetches()])
        return fetches[0], fetches[1:-1], fetches[-1]

    def _get_loss_inputs_dict(self, batch):
        # Override parent function to add seq_lens to tensor for additional LSTM
        loss_inputs = super(A3CPolicyGraph, self)._get_loss_inputs_dict(batch)
        loss_inputs[self.moa.seq_lens] = loss_inputs[self._seq_lens]
        return loss_inputs

    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        grads = tf.gradients(self._loss, self.var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        clipped_grads = list(zip(grads, self.var_list))
        return clipped_grads

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        """Extra values to fetch and return from compute_gradients()."""
        return {
            "stats": self.stats,
        }

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        """Extra values to fetch and return from compute_actions().

        By default we only return action probability info (if present).
        """
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self),
            **{"vf_preds": self.vf})

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        return sample_batch

    def compute_curiosity_reward(self, trajectory):
        """Compute curiosity reward of this agent
        """
        # TODO: B,N,A?
        # Predict the next state. (Shape is [B, N, A]??)
        true_logits, true_probs = self.predict_next_state(trajectory)

        # Logging curiosity metrics
        curiosity_per_agent = np.sum(curiosity_per_agent_step, axis=0)
        total_curiosity = np.sum(curiosity_per_agent_step)
        self.total_curiosity_reward.load(total_curiosity, session=self.sess)
        self.curiosity_per_agent = curiosity_per_agent

        # Summarize and clip influence reward
        curiosity = np.sum(curiosity_per_agent_step, axis=-1)
        curiosity = np.clip(curiosity, -self.aux_reward_clip,
                            self.aux_reward_clip)

        # Add to trajectory
        trajectory['rewards'] = trajectory['rewards'] + curiosity

        return trajectory
