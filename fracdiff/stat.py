from arch.unitroot import DFGLS

class StatTester:

    def __init__(self, method="ADF"):
        self.method = method

    @property
    def null_hypothesis(self) -> str:
        if self.method == "ADF":
            return "unit-root"

    def pvalue(self, x) -> float:
        """
        Return p-value of the stationarity test.

        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.

        Returns
        -------
        pvalue : float
            p-value of the stationarity test.
        """
        if self.method == "ADF":
            return DFGLS(x, trend='ct').pvalue

    def is_stat(self, x, pvalue=0.05) -> bool:
        """
        Return whether stationarity test implies stationarity.

        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.
        - pvalue : float, default 0.05
            Threshold of p-value.

        Note
        ----
        The name 'is_stat' may be misleading.
        Strictly speaking, `is_stat = True` implies that the null-hypothesis of
        the presence of a unit-root has been rejected (ADF test) or the null-hypothesis
        of the absence of a unit-root has not been rejected (KPSS test).

        Returns
        -------
        is_stat : bool
            True may imply the stationarity.
        """
        if self.null_hypothesis == "unit-root":
            return self.pvalue(x) < pvalue
