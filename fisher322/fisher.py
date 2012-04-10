""" Standalone implementation of Fisher's F CDF

From: http://www.netlib.org/toms/322

/*fisher.c - compute the two-tailed probability of correct rejection of the null
  hypothesis with an F-ratio of x, for m degrees of freedom in the numerator and
  n degrees of freedom in the denominator.  In the special case of only two
  populations, this is equivalent to Student's t-test with m=1 and x=t**2.
  Coded by Matthew Belmonte <mkb4@Cornell.edu>, 28 September 1995.  This
  implementation Copyright (c) 1995 by Matthew Belmonte.  Permission for use and
  distribution is hereby granted, subject to the restrictions that this
  copyright notice and reference list be included in its entirety, and that any
  and all changes made to the program be clearly noted in the program text.

  This software is provided 'as is', with no warranty, express or implied,
  including but not limited to warranties of merchantability or fitness for a
  particular purpose.  The user of this software assumes liability for any and
  all damages, whether direct or consequential, arising from its use.  The
  author of this implementation will not be liable for any such damages.

  References:

  Egon Dorrer, "Algorithm 322: F-Distribution [S14]", Communications of the
  Association for Computing Machinery 11:2:116-117 (1968).

  J.B.F. Field, "Certification of Algorithm 322 [S14] F-Distribution",
  Communications of the Association for Computing Machinery 12:1:39 (1969).

  Hubert Tolman, "Remark on Algorithm 322 [S14] F-Distribution", Communications
  of the Association for Computing Machinery 14:2:117 (1971).
*/
"""
import numpy as np


def f_sf(x, m, n):
    """ Fisher's F survival function (1-cdf)

    Returns p value for F statistic greater than or equal to `x`

    Using calling convention of scipy.stats.f.sf

    Parameters
    ----------
    x : array-like
        F statistic(s)
    m : int
        Degrees of freedom for the denominator
    m : int
        Degrees of freedom for the numerator

    Returns
    -------
    p : array
        probability of F value greater than or equal to `x`
    """
    return 1 - f_cdf(x, m, n)


def f_cdf(x, m, n):
    """ Fisher's F cumulative density function

    Returns p value for F statistic less than or equal to `x`

    Using calling convention of scipy.stats.f.cdf

    Parameters
    ----------
    x : array-like
        F statistic(s)
    m : int
        Degrees of freedom for the denominator
    m : int
        Degrees of freedom for the numerator

    Returns
    -------
    p : array
        probability of F value less than or equal to `x`
    """
    return _fisher(m, n, x)


def _fisher(m, n, x):
    """ Fisher's F cumulative density function

    This program is a very simple python port of fisher.c above. I have added
    the ability to deal with negative input, and x as a vector.

    Parameters
    ----------
    m : int
        Degrees of freedom for the denominator
    m : int
        Degrees of freedom for the numerator
    x : array-like
        F statistic

    Returns
    -------
    p : array
        probability of F value less than or equal to `x`
    """
    mf, nf = m, n
    m, n = np.round([mf, nf]).astype(int)
    if (mf, nf) != (m, n):
        raise ValueError("m, n need to be integers")
    # Negative values -> p == 0
    all_x = np.array(x, dtype=float)
    is_scalar = all_x.ndim == 0
    if is_scalar:
        if all_x <= 0:
            return 0
        all_x = np.atleast_1d(all_x)
    all_p = np.zeros_like(all_x)
    gt0 = all_x > 0
    finites = all_x != np.inf
    all_p[-finites] = 1
    valid = gt0 & finites
    x = all_x[valid]
    a = 2*(m//2)-m+2;
    b = 2*(n//2)-n+2;
    w = (x*m)/n;
    z = 1.0/(1.0+w);
    if (a == 1):
        if (b == 1):
            p = np.sqrt(w);
            y = 0.3183098862;
            d = y*z/p;
            p = 2.0*y*np.arctan(p);
        else:
            p = np.sqrt(w*z);
            d = 0.5*p*z/w;
    elif (b == 1):
        p = np.sqrt(z);
        d = 0.5*z*p;
        p = 1.0-p;
    else:
        d = z*z;
        p = w*z;
    y = 2.0*w/z;
    if (a == 1):
        for j in range(b+2, n+1, 2):
            d *= (1.0+1.0/(j-2))*z;
            p += d*y/(j-1);
    else:
        zk = z ** float((n-1)//2)
        d *= (zk*n)/b;
        p = p*zk+w*z*(zk-1.0)/(z-1.0);
    y = w*z;
    z = 2.0/z;
    b = n-2;
    for i in range(a+2, m+1, 2):
        j = i+b;
        d *= (y*j)/(i-2);
        p -= z*d/j;
    all_p[valid] = p
    if is_scalar:
        return np.asscalar(all_p)
    return all_p
