"""Focused metadata regression checks using the actually rendered trial records.

The production render verified full bytes. These isolated negative metadata
tests reuse that observed hash map to avoid re-reading gigabytes per mutation.
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("figure_adapter",ROOT/"render_comparison.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)
RUN = ROOT/"outputs/family1_explicit_trial_v1"
SCORE = ROOT/"scores/family1_explicit_trial_v1"
FIG = ROOT/"figures/family1_explicit_trial_v1_20261019_v1"
OBSERVED_HASHES = ADAPTER.read(FIG/"input_hashes_before.json")


class MetadataChecks(unittest.TestCase):
    def prepare(self, changes=None, name="family1_explicit_trial_v1"):
        original_read, original_sha = ADAPTER.read, ADAPTER.sha
        changes = changes or {}

        def read(path):
            value = original_read(path)
            change = changes.get(str(Path(path)))
            if change:
                value = copy.deepcopy(value)
                change(value)
            return value

        def sha(path):
            key = str(Path(path).resolve())
            return OBSERVED_HASHES[key] if key in OBSERVED_HASHES else original_sha(path)

        with patch.object(ADAPTER,"read",side_effect=read),patch.object(ADAPTER,"sha",side_effect=sha):
            return ADAPTER.prepare(RUN,SCORE,name)

    def test_observed_valid_bindings(self):
        context = self.prepare()
        self.assertEqual(context["counts"]["passing_planes_both_scans"],0)
        self.assertEqual(context["counts"]["passing_planes_by_scan"],{"pre":0,"post":1})

    def test_wrong_candidate_label_refused(self):
        with self.assertRaisesRegex(ValueError,"--name"):
            self.prepare(name="unrelated_candidate")

    def test_completion_export_digest_mismatch_refused(self):
        def mutation(value):
            value["artifacts"][str(RUN/"candidate_pre_complex.npy")] = "0"*64
        with self.assertRaisesRegex(ValueError,"hashes disagree"):
            self.prepare({str(RUN/"completed.json"):mutation})

    def test_completed_update_count_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError,"update count mismatch"):
            self.prepare({str(RUN/"metrics.json"):lambda value:value.update(steps_observed=499)})

    def test_whole_contract_volume_swap_refused(self):
        def mutation(value):
            value["opponents"]["ZS-SSDU"]["volumes"]["pre"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"manifests disagree"):
            self.prepare({str(SCORE/"whole_head_contract.json"):mutation})

    def test_plane_count_tamper_refused(self):
        with self.assertRaisesRegex(ValueError,"summary counts disagree"):
            self.prepare({str(SCORE/"lesion_planes.json"):lambda value:value.update(passing_planes_both_scans=59)})

    def test_frozen_opponent_plane_tamper_refused(self):
        def mutation(value):
            value["planes"][0]["comparisons"]["pre:psnr"]["opponents"]["ZS-SSDU"] += 1
        with self.assertRaisesRegex(ValueError,"opponent plane scores changed"):
            self.prepare({str(SCORE/"lesion_planes.json"):mutation})

    def test_plane_identity_tamper_refused(self):
        def mutation(value):
            value["planes"][0]["index"] += 1
        with self.assertRaisesRegex(ValueError,"plane identities changed"):
            self.prepare({str(SCORE/"lesion_planes.json"):mutation})

    def test_false_raw_win_refused(self):
        def mutation(value):
            value["planes"][0]["comparisons"]["pre:psnr"]["strict_win"] = True
        with self.assertRaisesRegex(ValueError,"raw plane decision"):
            self.prepare({str(SCORE/"lesion_planes.json"):mutation})


if __name__ == "__main__":
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetadataChecks))
    print(log.getvalue())
    record = {"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
              "successful":result.wasSuccessful(),"test_kind":"Metadata mutation regression checks; observed full-byte hashes reused",
              "source_sha256":hashlib.sha256((ROOT/"render_comparison.py").read_bytes()).hexdigest(),
              "observed_full_byte_verification":str(FIG/"provenance.json"),"log":log.getvalue()}
    with (ROOT/"verification/figure_adapter_checks_v1.json").open("x") as stream:
        json.dump(record,stream,indent=2)
        stream.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
