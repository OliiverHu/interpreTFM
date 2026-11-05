# Security policy

interpreTFM is research software that runs locally on data you provide; it has no hosted service.

## Supported versions

Security fixes are made for the latest release only.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting ("Report a vulnerability" under the repository's Security tab) or contact the maintainer through the details on the maintainer's GitHub profile. Include a description, the affected version and steps to reproduce. Reports are acknowledged as soon as possible.

## Data protection

Do not share patient-level or otherwise restricted data in issues, pull requests or discussions. The package loads model checkpoints with `torch.load`, which can execute code from untrusted files: load only checkpoints from sources you trust.
