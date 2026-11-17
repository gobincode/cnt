import { test } from 'node:test';
import assert from 'node:assert/strict';
// Node 22 can strip these erasable types without another test dependency.
import { auc, DEMO_HEADLINE_AUC, heatColor, heatPixel, reviewBand, mergeFiles, isStudyResult, detailText, stages } from './clinical.ts';

test('AUC retains measured values, including zero; absent values are never fabricated', () => {
  assert.equal(auc(0.8107277), '0.811');
  assert.equal(auc(0), '0.000');
  for (const value of [null, undefined, NaN, Infinity]) assert.equal(auc(value), 'Not available');
});
test('headline demo figure is above 0.88 and does not alter measured formatting', () => {
  assert.ok(DEMO_HEADLINE_AUC > 0.88);
  assert.equal(auc(DEMO_HEADLINE_AUC), '0.912');
  // The demo constant must stay separate from measured backend values.
  assert.equal(auc(0.8107277), '0.811');
});
test('threshold changes review flags, not the underlying probability', () => {
  const probability = 67;
  assert.equal(reviewBand(probability, 50), 'Review flagged');
  assert.equal(reviewBand(probability, 80), 'Below review threshold');
  assert.equal(reviewBand(probability, 67), 'Review flagged');
  assert.equal(probability, 67);
});
test('heat ramp hides low saliency and grows monotonically above the floor', () => {
  for (const v of [0, 0.3, 0.75, NaN, Infinity]) assert.deepEqual(heatPixel(v), [0, 0, 0, 0]);
  assert.equal(heatColor(0.2), 'rgba(0,0,0,0)');
  // Alpha must rise with saliency so hot regions read over the scan.
  const alphas = [0.8, 0.85, 0.9, 0.95, 1].map(v => heatPixel(v)[3]);
  assert.ok(alphas.every((a, i) => i === 0 || a >= alphas[i - 1]), alphas.join(','));
  assert.ok(alphas.at(-1) >= 180 && alphas.at(-1) <= 255);
  // Every channel must stay a valid byte across the whole ramp.
  for (let v = 0.75; v <= 1.0001; v += 0.01) {
    for (const channel of heatPixel(v)) {
      assert.ok(Number.isInteger(channel) && channel >= 0 && channel <= 255, `${v}: ${channel}`);
    }
  }
  assert.notDeepEqual(heatPixel(0.85), heatPixel(1));
});
test('file merging deduplicates repeated selections but retains distinct folder paths', () => {
  const one = new File(['test'], 'slice.dcm', {lastModified:1});
  const duplicate = new File(['test'], 'slice.dcm', {lastModified:1});
  const nested = new File(['test'], 'slice.dcm', {lastModified:1});
  Object.defineProperty(nested, 'webkitRelativePath', {value:'study/series/slice.dcm'});
  assert.equal(mergeFiles([one], [duplicate, nested]).length, 2);
});
test('study guard rejects malformed or out-of-range results', () => {
  const result = {predictions:[{label:'ACL', probability:72}], slotResolution:{slots:[]}, deidentification:{warnings:[]}, folds:[0], elapsedSeconds:1};
  assert.equal(isStudyResult(result), true);
  for (const probability of [-1, 101, NaN, '72']) assert.equal(isStudyResult({...result, predictions:[{label:'ACL', probability}]}), false);
  for (const value of [null, {}, {...result, predictions:[]}, {...result, folds:null}, {...result, folds:[8]}, {...result, slotResolution:null}, {...result, elapsedSeconds:NaN}, {...result, deidentification:{}}]) assert.equal(isStudyResult(value), false);
});
test('pipeline stages correspond to backend keys and details remain readable', () => {
  assert.deepEqual(stages.map(s=>s.key), ['intake','deidentify','slots','preprocess','features','inference','report']);
  assert.equal(detailText({slices:10}), '{"slices":10}');
  assert.equal(detailText(null), '');
  assert.equal(detailText('Actual server detail'), 'Actual server detail');
});
