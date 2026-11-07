import unittest
from desktop_detector.core import (Detection, Region, ResponseSchemaError, SingleFlight, InferenceSettings,
                                   INTERVAL_CHOICES, MODE_BINARY, fit_image, validate_interval,
                                   make_questions, parse_boxes, validate_threshold, make_binary_question,
                                   parse_binary, validate_categories, make_choice_question, parse_choice)


def response(boxes):
    return {"answers": {"objects": {"boxes": boxes, "count": len(boxes)}}}


def box(**extra):
    result = {"box": {"left": .1, "top": .2, "right": .6, "bottom": .8}, "confidence": .9, "found": .99}
    result.update(extra)
    return result


class GeometryTests(unittest.TestCase):
    def test_adjacent_not_overlap(self):
        self.assertFalse(Region(0, 0, 100, 100).overlaps(Region(100, 0, 50, 50)))
    def test_partial_overlap(self):
        self.assertTrue(Region(0, 0, 100, 100).overlaps(Region(99, 99, 50, 50)))
    def test_contains_overlap(self):
        self.assertTrue(Region(0, 0, 100, 100).overlaps(Region(20, 30, 10, 10)))
    def test_inside(self):
        self.assertTrue(Region(0, 0, 200, 100).inside(200, 100))
        self.assertFalse(Region(-1, 0, 200, 100).inside(200, 100))
        self.assertFalse(Region(1, 0, 200, 100).inside(200, 100))
    def test_invalid_region(self):
        with self.assertRaises(ValueError): Region(0, 0, 0, 10)
    def test_letterbox(self):
        self.assertEqual(fit_image(1920, 1080, 400, 300), (400, 225, 0, 37))
    def test_portrait(self):
        self.assertEqual(fit_image(100, 200, 400, 300), (150, 300, 125, 0))
    def test_high_dpi_normalized_mapping(self):
        self.assertEqual(Detection(.25, .25, .75, .75).pixels(400, 200), (100, 50, 300, 150))
        self.assertEqual(fit_image(2560, 1440, 640, 360), (640, 360, 0, 0))


class ParsingTests(unittest.TestCase):
    def test_inference_settings_validate_mode_inputs(self):
        settings = InferenceSettings(MODE_BINARY, "魚が映っているか？", .2, 8, 768,
                                     "映っている魚をすべて囲む")
        self.assertEqual(settings.threshold, .2)
        with self.assertRaises(ValueError):
            InferenceSettings(MODE_BINARY, "", .2, 8, 768)
    def test_binary_question_and_score(self):
        self.assertEqual(make_binary_question(" fish ")["decision"]["type"], "noul")
        self.assertEqual(parse_binary({"answers": {"decision": {"type": "noul", "noul": .28}}}), .28)
        with self.assertRaises(ResponseSchemaError): parse_binary({"answers": {"decision": {"noul": 2}}})
    def test_choice_includes_none_and_selects_highest_score(self):
        categories = validate_categories("魚がある\n女性が映っている\n")
        question = make_choice_question(categories)["category"]
        self.assertEqual(question["criteria"]["none"], "上記のどれにも該当しない")
        answer = {"answers": {"category": {"choice": "c0", "probabilities": {"c0": .3, "c1": .6, "none": .1}}}}
        self.assertEqual(parse_choice(answer, categories)[:2], ("女性が映っている", .6))
        with self.assertRaises(ValueError): validate_categories("魚がある\n該当しない")
    def test_threshold_validation(self):
        self.assertEqual(validate_threshold("0.65"), .65)
        for value in ("", "abc", -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): validate_threshold(value)
    def test_request(self):
        self.assertEqual(make_questions(" fish ", 4), {"objects": {"type": "boxes", "instructions": "fish", "max": 4}})
    def test_empty_prompt(self):
        with self.assertRaises(ValueError): make_questions("   ")
    def test_maximum(self):
        for n in (0, 17, 1.5, True):
            with self.assertRaises(ValueError): make_questions("fish", n)
    def test_empty_boxes(self):
        self.assertEqual(parse_boxes(response([])), [])
    def test_nested_box(self):
        ds = parse_boxes(response([box()]))
        self.assertEqual(ds[0].pixels(1000, 500), (100, 100, 600, 400))
    def test_direct_box(self):
        ds = parse_boxes(response([{"left": .1, "top": .2, "right": .8, "bottom": .9}]))
        self.assertIsNone(ds[0].confidence)
    def test_filter_confidence(self):
        self.assertEqual(parse_boxes(response([box(confidence=.3)]), .5), [])
    def test_missing_score_not_made_up(self):
        ds = parse_boxes(response([box(confidence=None)]), .5)
        self.assertIsNone(ds[0].confidence)
    def test_found_threshold(self):
        self.assertEqual(parse_boxes(response([box(found=.1)])), [])
    def test_null_box(self):
        self.assertEqual(parse_boxes(response([box(box=None)])), [])
    def test_tiny_outside_clamped(self):
        ds = parse_boxes(response([box(box={"left": -.001, "top": .1, "right": 1.001, "bottom": .9})]))
        self.assertEqual((ds[0].left, ds[0].right), (0, 1))
    def test_pixel_coordinates_rejected(self):
        with self.assertRaises(ResponseSchemaError):
            parse_boxes(response([box(box={"left": 10, "top": 20, "right": 100, "bottom": 200})]))
    def test_reversed_rejected(self):
        with self.assertRaises(ResponseSchemaError):
            parse_boxes(response([box(box={"left": .8, "top": .1, "right": .2, "bottom": .9})]))
    def test_nan_rejected(self):
        with self.assertRaises(ResponseSchemaError):
            parse_boxes(response([box(box={"left": float('nan'), "top": .1, "right": .8, "bottom": .9})]))
    def test_bad_schema_not_silent_zero(self):
        for raw in ({}, {"answers": {}}, {"answers": {"objects": {}}}, response([{"garbage": 1}])):
            with self.assertRaises(ResponseSchemaError): parse_boxes(raw)
    def test_bad_probability(self):
        with self.assertRaises(ResponseSchemaError): parse_boxes(response([box(confidence=90)]))


class SchedulingTests(unittest.TestCase):
    def test_interval_choices_include_slow_cpu_settings(self):
        self.assertEqual(INTERVAL_CHOICES, ("0.5", "1.0", "2.0", "5.0", "10.0"))
        self.assertEqual(validate_interval("5.0"), 5.0)
        self.assertEqual(validate_interval("10.0"), 10.0)
        for value in ("", "nan", "0", "61"):
            with self.assertRaises(ValueError): validate_interval(value)
    def test_one_inflight(self):
        g = SingleFlight(); g.start(.5, 0)
        self.assertIsNotNone(g.claim(0))
        self.assertIsNone(g.claim(10))
    def test_fast_inference_waits(self):
        g = SingleFlight(); g.start(.5, 0); token = g.claim(0)
        self.assertTrue(g.finish(token)); self.assertIsNone(g.claim(.2))
        self.assertIsNotNone(g.claim(.5))
    def test_slow_inference_no_catchup_burst(self):
        g = SingleFlight(); g.start(.5, 0); token = g.claim(0)
        self.assertTrue(g.finish(token)); self.assertIsNotNone(g.claim(1.4))
        self.assertAlmostEqual(g.next_due, 1.9)
    def test_stop_discards_late_result(self):
        g = SingleFlight(); g.start(.5, 0); token = g.claim(0); g.stop()
        self.assertFalse(g.finish(token)); self.assertIsNone(g.claim(10))
    def test_cannot_restart_before_pending_finishes(self):
        g = SingleFlight(); g.start(.5, 0); token = g.claim(0); g.stop()
        with self.assertRaises(RuntimeError): g.start(.5, 1)
        g.finish(token); g.start(.5, 2); self.assertIsNotNone(g.claim(2))
    def test_interval_validation(self):
        for n in (0, -1, float("nan")):
            with self.assertRaises(ValueError): SingleFlight().start(n, 0)


if __name__ == "__main__":
    unittest.main()
