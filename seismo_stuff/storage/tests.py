def test_stellar_parameters():
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    s = lc.to_periodogram().estimate_snr()
    test = s.estimate_stellar_parameters(6000.)
    assert np.isreal(test)
