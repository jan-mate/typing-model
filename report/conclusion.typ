= Conclusion
Machine learning models predict typing speed from keystroke data. Nonlinear models outperform linear ones. Including future context improves QWERTY accuracy. This confirms typing is anticipatory. The models also generalize from QWERTY to Dvorak, the first zero-shot evaluation of a typing-speed model on a different layout. LightGBM transfers best. As the deployed ensemble, it reaches the lowest Dvorak MAE (0.588) and the least bias shift. This makes it the most trustworthy across layouts. MLP Main scores best on QWERTY (0.566). But its edge vanishes on Dvorak. There the near-raw-coordinate MLP DL nearly matches it, a sign MLP Main overfits.

Whether the model can guide layout optimization remains unclear. Its bias on layouts beyond QWERTY and Dvorak is unmeasured. LightGBM is the least biased model, but also the slowest. Scoring the many candidate layouts optimization requires is computationally impractical.

The models quantify the speed benefits of typing system optimizations. One-shot shift, on the displaced semicolon slot, saves 18.5 ms per isolated capital, a 0.71% speedup, or about 1.1% if the benefit extends to sentence-initial capitals. A repeat key in that slot gives a 0.26% gain. Abbreviation dictionaries increase speed by up to 12.99% for a 160-entry dictionary, an estimate that assumes instant recall of each mapping. After adjusting for layout bias, Dvorak is 0.7% faster than QWERTY.

Two directions would improve the models. The first is data: only QWERTY and Dvorak have keystroke records. More layouts would lower the MAE, shrink the bias shift, and tighten the confidence intervals. The second is architecture: simpler window models and fewer features would likely reduce the overfitting that MLP Main shows.

