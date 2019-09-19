# Speech Enhancement with Mimic Loss

This project seeks to bring together the surprisingly separate worlds of
speech enhancement and noise-robust ASR by applying phonetic knowledge to
improve a front-end enhancement module. This can improve both intellegibility
metrics (STOI and eSTOI) as well as ASR performed on the outputs of this
enhancement system.

The backbone of this project is work by Pandey et al. [1] which performs
denoising in the time domain, but generates a loss in the spectral domain
and backpropagates through the STFT to improve the denoising model. We
apply mimic loss [2] in the spectral domain and backpropagate to the
time domain.

[1] Ashutosh Pandey and DeLiang Wang, "A new framework for supervised
speech enhancement in the time domain," Interspeech, 2018.

[2] Deblin Bagchi, Peter Plantinga, Adam Stiff, and Eric Fosler-Lussier,
"Spectral feature mapping with mimic loss for robust ASR," ICASSP, 2018
