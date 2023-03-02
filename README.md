# asianoptions
Pricing Asian call options using GBM Black-Scholes.
Asian call options are priced with regards to the average price of their underlying asset. 

The hypothesis tested here, is that the price of the options evolve as:

$$ e^{-rT}\mathbb{E}\left[ \frac{1}{T}\int_0^T S(t) dt - K \right ] + \mathcal{O}(N^{-\alpha}), $$

where $r$ is the short rate, $T$ the maturity, $S(t)$ a GBM, $K$ the strike price and $N$ the sampling rate of the underlying stock price. 
