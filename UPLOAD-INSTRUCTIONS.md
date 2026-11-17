# Ready-to-upload course studies

In the UI, choose **Study intake → Choose folder**, then select ONE of:

* `upload-studies/01-complete-right-synthetic` — right side, six slots, 72 DICOM slices.
* `upload-studies/02-complete-left-synthetic` — left side, six slots, 72 DICOM slices.
* `upload-studies/03-partial-right-synthetic` — right side, three slots, 36 DICOM slices.

Choose the whole study folder, not an individual series. Its nested folders are
picked up together. Do NOT select the parent `upload-studies` folder: that mixes
three studies plus a manifest, which the API rejects. Clear the file selection
before choosing another study. Start with fold 0 for speed.

All images are small, computer-generated analytic phantoms. They are not actual
MRI anatomy or labelled medical data. Real inference still runs, but its outputs
on these synthetic inputs have no diagnostic meaning and cannot measure accuracy.

The UI's **91.2% headline demo AUC** banner is presentation content for the demo,
not measured performance. Genuine benchmark values and federated results are
displayed unchanged beside it.

Regenerate into a new/empty location using `server/generate_upload_studies.py`.
The generator refuses to overwrite existing study folders.
