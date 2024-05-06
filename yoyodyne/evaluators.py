"""Evaluators."""

from __future__ import annotations
import abc
import argparse
import dataclasses
from typing import List, Optional

import numpy
import torch
from torch.nn import functional

# from torchmetrics.text import CharErrorRate
from torchmetrics.functional.text.helper import _edit_distance

from . import defaults


class Error(Exception):
    pass


@dataclasses.dataclass
class EvalItem:
    per_sample_metrics: List[float]

    @property
    def metric(self) -> float:
        """Computes the micro-average of the metric."""
        return numpy.mean(self.per_sample_metrics)

    def __add__(self, other_eval: EvalItem) -> EvalItem:
        """Adds two EvalItem by concatenating the list of individual metrics.

        Args:
            other_eval (EvalItem): The other eval item to add to self.

        Returns:
            EvalItem.
        """
        return EvalItem(
            self.per_sample_metrics + other_eval.per_sample_metrics
        )


class Evaluator(abc.ABC):
    """Evaluator interface."""

    def evaluate(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        k_golds: Optional[int]=None
    ) -> EvalItem:
        """Computes the evaluation metric.

        This is the top-level public method that should be called by
        evaluating code.

        Args:
            predictions (torch.Tensor): B x vocab_size x seq_len.
            golds (torch.Tensor): B x seq_len x 1.
            end_idx (int): end of sequence index.
            pad_idx (int): padding index.

        Returns:
            EvalItem.
        """
        if predictions.size(0) != golds.size(0):
            raise Error(
                f"Preds batch size ({predictions.size(0)}) and "
                f"golds batch size ({golds.size(0)} do not match"
            )
        # Gets the max value at each dim2 in predictions.
        _, predictions = torch.max(predictions, dim=2)
        # Finalizes the predictions.
        predictions = self.finalize_predictions(predictions, end_idx, pad_idx)
        golds = self.finalize_golds(golds, end_idx, pad_idx, k=k_golds)
        return self.get_eval_item(predictions, golds, pad_idx)

    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        raise NotImplementedError

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def finalize_golds(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        k: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class AccuracyEvaluator(Evaluator):
    """Evaluates accuracy."""

    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        if predictions.size(1) > golds.size(1):
            predictions = predictions[:, : golds.size(1)]
        elif predictions.size(1) < golds.size(1):
            num_pads = (0, golds.size(1) - predictions.size(1))
            predictions = functional.pad(
                predictions, num_pads, "constant", pad_idx
            )
        # Gets the count of exactly matching tensors in the batch.
        # -> B.
        accs = (predictions.to(golds.device) == golds).all(dim=1).tolist()
        return EvalItem(accs)

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        """Finalizes predictions.

        Cuts off tensors at the first end_idx, and replaces the rest of the
        predictions with pad_idx, as these are erroneously decoded while the
        rest of the batch is finishing decoding.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            torch.Tensor: finalized predictions.
        """
        # Not necessary if batch size is 1.
        if predictions.size(0) == 1:
            return predictions
        for i, prediction in enumerate(predictions):
            # Gets first instance of EOS.
            eos = (prediction == end_idx).nonzero(as_tuple=False)
            if len(eos) > 0 and eos[0].item() < len(prediction):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                eos = eos[0]
            else:
                # Leaves predictions[i] alone.
                continue
            # Hack in case the first prediction is EOS. In this case
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            eos[eos == 0] = 1
            symbols, *_ = torch.split(prediction, eos)
            # Replaces everything after with PAD, to replace erroneous decoding
            # While waiting on the entire batch to finish.
            pads = (
                torch.ones(
                    len(prediction) - len(symbols), device=symbols.device
                )
                * pad_idx
            )
            pads[0] = end_idx
            # Makes an in-place update to an inference tensor.
            with torch.inference_mode():
                predictions[i] = torch.cat((symbols, pads))
        return predictions

    def finalize_golds(
        self,
        golds: torch.Tensor,
        *args,
        **kwargs,
    ):
        return golds

    @property
    def name(self) -> str:
        return "accuracy"


class SEREvaluator(Evaluator):
    """Evaluates symbol error rate.

    Here, a symbol is defined by the user specified tokenization."""

    def _compute_ser(
        self,
        preds: List[str],
        target: List[str],
    ) -> float:
        errors = _edit_distance(preds, target)
        total = len(target)
        return errors / total

    def get_eval_item(
        self,
        predictions: List[List[str]],
        golds: List[List[str]],
        pad_idx: int,
    ) -> EvalItem:
        sers = [self._compute_ser(p, g) for p, g in zip(predictions, golds)]
        return EvalItem(sers)

    def _finalize_tensor(
        self,
        tensor: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> List[List[str]]:
        # Not necessary if batch size is 1.
        if tensor.size(0) == 1:
            # Returns a list of a numpy char vector.
            # This is allows evaluation over strings without converting
            # integer indices back to symbols.
            return [numpy.char.mod("%d", tensor.cpu().numpy())]
        out = []
        for prediction in tensor:
            # Gets first instance of EOS.
            eos = (prediction == end_idx).nonzero(as_tuple=False)
            if len(eos) > 0 and eos[0].item() < len(prediction):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                eos = eos[0]
            else:
                # Leaves tensor[i] alone.
                out.append(numpy.char.mod("%d", prediction))
                continue
            # Hack in case the first prediction is EOS. In this case
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            eos[eos == 0] = 1
            symbols, *_ = torch.split(prediction, eos)
            # Accumulates a list of numpy char vectors.
            out.append(numpy.char.mod("%d", symbols))
        return out

    def finalize_predictions(
        self,
        predictions: torch.Tensor,
        end_idx: int,
        pad_idx: int,
    ) -> List[List[str]]:
        """Finalizes predictions.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            torch.Tensor: finalized predictions.
        """
        return self._finalize_tensor(predictions, end_idx, pad_idx)

    def finalize_golds(
        self,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        *args,
        **kwargs,
    ):
        return self._finalize_tensor(golds, end_idx, pad_idx)

    @property
    def name(self) -> str:
        return "ser"


class AccuracyInTop1Evaluator(AccuracyEvaluator):
    
    def get_eval_item(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        """Creates an EvalItem with the AccuracyInTop1.

        Accuracy in top 1 compares the top system prediction to all k gold
        standard answers. This is essentially the same as exact match accuracy,
        but we return one if ANY of the k golds is an exact match.

        Args:
            predictions (torch.Tensor): System predictions. B x seq_len.
            golds (torch.Tensor): Gold sequences. B x k x seq_len.
            pad_idx (int).

        Returns:
            EvalItem.
        """
        if predictions.size(1) > golds.size(2):
            predictions = predictions[:, : golds.size(2)]
        elif predictions.size(1) < golds.size(2):
            num_pads = (0, golds.size(2) - predictions.size(1))
            predictions = functional.pad(
                predictions, num_pads, "constant", pad_idx
            )
        # Compares exact match of each prediction to all k possible golds
        # by repeating the prediction k times.
        matches = (
            predictions.to(golds.device).unsqueeze(1).repeat(1, golds.size(1), 1) == golds
        ).all(dim=2)
        # Checks if each prediction in the batch matches any of the k golds.
        top_1_accs = matches.any(dim=1).tolist()
        return EvalItem(top_1_accs)
    
    def finalize_golds(
        self,
        golds: torch.Tensor,
        end_idx: int,
        pad_idx: int,
        k: int,
    ):
        return golds.reshape(-1, k, int(golds.size(1) / k))

    @property
    def name(self):
        return "accuracy_in_top_1"


_eval_factory = {
    "accuracy": AccuracyEvaluator,
    "ser": SEREvaluator,
    "accuracy_in_top_1": AccuracyInTop1Evaluator,
}


def get_evaluator(eval_metric: str) -> Evaluator:
    """Gets the requested Evaluator given the specified metric.

    Args:
        eval_metric (str).

    Raises:
        Error.

    Returns:
        Evaluator.
    """
    try:
        return _eval_factory[eval_metric]
    except KeyError(eval_metric):
        raise Error(f"No evaluation metric {eval_metric}")


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds LSTM configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    parser.add_argument(
        "--eval_metric",
        action="append",
        choices=_eval_factory.keys(),
        default=defaults.EVAL_METRICS,
        help="Which evaluation metrics to use. Default: %(default)s.",
    )
