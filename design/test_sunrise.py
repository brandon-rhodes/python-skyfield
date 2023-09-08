#!/usr/bin/env python
#
# The new 'rising' routine (a) takes only a few steps before returning
# its answer, and (b) ignores the real angular velocity of the target
# across the sky.  It should therefore perform worst in the case of the
# Moon.  Is it able to match USNO predictions for one year?

import datetime as dt
from skyfield import almanac
from skyfield.api import load, wgs84

# From:
# https://aa.usno.navy.mil/calculated/rstt/year?ID=AA&year=2023&task=0&lat=36.95&lon=-112.52&label=Fredonia%2C+AZ&tz=7&tz_sign=-1&submit=Get+Data

TABLE = """\
             o  ,    o  ,                                    FREDONIA, AZ                              Astronomical Applications Dept.
Location: W112 31, N36 57                          Rise and Set for the Sun for 2023                   U. S. Naval Observatory        
                                                                                                       Washington, DC  20392-5420     
                                                      Zone:  7h West of Greenwich                                                     
                                                                                                                                      
                                                                                                                                      
       Jan.       Feb.       Mar.       Apr.       May        June       July       Aug.       Sept.      Oct.       Nov.       Dec.  
Day Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set  Rise  Set
     h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m   h m  h m
01  0743 1724  0733 1755  0701 1824  0616 1852  0536 1919  0512 1944  0514 1954  0535 1937  0601 1859  0625 1814  0654 1733  0724 1714
02  0743 1725  0732 1756  0700 1825  0615 1853  0535 1920  0512 1945  0515 1953  0536 1936  0602 1857  0626 1812  0655 1732  0725 1713
03  0744 1726  0731 1757  0658 1826  0613 1854  0534 1921  0512 1945  0515 1953  0537 1935  0602 1856  0627 1811  0656 1731  0726 1713
04  0744 1727  0730 1758  0657 1827  0612 1855  0533 1922  0511 1946  0516 1953  0538 1934  0603 1854  0628 1809  0657 1730  0727 1713
05  0744 1728  0729 1759  0656 1828  0610 1856  0532 1922  0511 1946  0516 1953  0539 1933  0604 1853  0629 1808  0658 1729  0728 1713
06  0744 1728  0728 1801  0654 1829  0609 1857  0531 1923  0511 1947  0517 1953  0539 1932  0605 1851  0630 1806  0659 1728  0729 1713
07  0744 1729  0727 1802  0653 1830  0607 1858  0530 1924  0511 1948  0517 1953  0540 1931  0606 1850  0630 1805  0700 1727  0730 1713
08  0744 1730  0726 1803  0651 1831  0606 1859  0529 1925  0510 1948  0518 1952  0541 1930  0607 1848  0631 1803  0701 1726  0731 1713
09  0743 1731  0725 1804  0650 1832  0604 1859  0528 1926  0510 1949  0519 1952  0542 1929  0607 1847  0632 1802  0702 1725  0731 1713
10  0743 1732  0724 1805  0648 1833  0603 1900  0527 1927  0510 1949  0519 1952  0543 1928  0608 1845  0633 1800  0703 1724  0732 1713
11  0743 1733  0723 1806  0647 1834  0602 1901  0526 1928  0510 1950  0520 1951  0544 1926  0609 1844  0634 1759  0704 1724  0733 1714
12  0743 1734  0722 1807  0646 1835  0600 1902  0525 1929  0510 1950  0520 1951  0544 1925  0610 1842  0635 1758  0705 1723  0734 1714
13  0743 1735  0721 1808  0644 1835  0559 1903  0524 1929  0510 1950  0521 1950  0545 1924  0611 1841  0636 1756  0706 1722  0734 1714
14  0743 1736  0720 1809  0643 1836  0557 1904  0523 1930  0510 1951  0522 1950  0546 1923  0611 1839  0637 1755  0707 1721  0735 1714
15  0742 1737  0719 1810  0641 1837  0556 1905  0522 1931  0510 1951  0522 1949  0547 1922  0612 1838  0638 1754  0708 1721  0736 1715
16  0742 1738  0718 1811  0640 1838  0555 1906  0521 1932  0510 1952  0523 1949  0548 1920  0613 1836  0639 1752  0709 1720  0736 1715
17  0742 1739  0716 1812  0638 1839  0553 1906  0521 1933  0510 1952  0524 1948  0549 1919  0614 1835  0639 1751  0710 1719  0737 1715
18  0741 1740  0715 1813  0637 1840  0552 1907  0520 1934  0510 1952  0525 1948  0549 1918  0615 1833  0640 1750  0712 1719  0738 1716
19  0741 1741  0714 1814  0635 1841  0551 1908  0519 1934  0511 1952  0525 1947  0550 1917  0615 1832  0641 1748  0713 1718  0738 1716
20  0740 1742  0713 1815  0634 1842  0549 1909  0518 1935  0511 1953  0526 1947  0551 1915  0616 1830  0642 1747  0714 1717  0739 1716
21  0740 1743  0711 1816  0632 1843  0548 1910  0518 1936  0511 1953  0527 1946  0552 1914  0617 1829  0643 1746  0715 1717  0739 1717
22  0739 1744  0710 1817  0631 1844  0547 1911  0517 1937  0511 1953  0527 1945  0553 1913  0618 1827  0644 1744  0716 1716  0740 1717
23  0739 1745  0709 1818  0629 1844  0546 1912  0517 1938  0511 1953  0528 1945  0554 1911  0619 1826  0645 1743  0717 1716  0740 1718
24  0738 1746  0708 1819  0628 1845  0544 1913  0516 1938  0512 1953  0529 1944  0554 1910  0619 1824  0646 1742  0718 1716  0741 1719
25  0738 1748  0706 1820  0626 1846  0543 1914  0515 1939  0512 1954  0530 1943  0555 1909  0620 1823  0647 1741  0719 1715  0741 1719
26  0737 1749  0705 1821  0625 1847  0542 1914  0515 1940  0512 1954  0531 1942  0556 1907  0621 1821  0648 1740  0720 1715  0742 1720
27  0736 1750  0704 1822  0623 1848  0541 1915  0514 1941  0513 1954  0531 1941  0557 1906  0622 1820  0649 1738  0721 1715  0742 1720
28  0736 1751  0702 1823  0622 1849  0539 1916  0514 1941  0513 1954  0532 1941  0558 1904  0623 1818  0650 1737  0722 1714  0742 1721
29  0735 1752             0620 1850  0538 1917  0513 1942  0513 1954  0533 1940  0558 1903  0624 1817  0651 1736  0723 1714  0743 1722
30  0734 1753             0619 1851  0537 1918  0513 1943  0514 1954  0534 1939  0559 1902  0624 1815  0652 1735  0723 1714  0743 1722
31  0733 1754             0617 1852             0513 1943             0535 1938  0600 1900             0653 1734             0743 1723

Add one hour for daylight time, if and when in use.
"""

def main():
    usno_rises = []  # (month, minutes_into_day)

    for line in TABLE.splitlines():
        if not line[0:1].isdigit():
            continue
        day_of_month = int(line[:2])
        months_and_texts = [(i+1, line[4+i*11:8+i*11]) for i in range(0, 11)]
        for month, text in months_and_texts:
            if text.isdigit():
                hours = int(text[0:2])
                minutes = int(text[2:4])
                usno_rises.append((month, day_of_month, hours * 60 + minutes))

    usno_rises.sort()
    #print(usno_rises)

    ts = load.timescale()
    t0 = ts.utc(2023, 1, 1, 7)
    #t0 = ts.utc(2023, 1, 1, 10)
    t1 = ts.utc(2024, 1, 1, 7)
    #t1 = ts.utc(2023, 4, 1, 7)
    e = load('de421.bsp')
    #e = load('de430.bsp')
    fredonia = wgs84.latlon(36 + 57/60.0, - (112 + 31/60.0))
    observer = e['Earth'] + fredonia
    horizon = -0.8333
    #horizon = -0.8333333333333
    t, y = almanac.find_risings(observer, e['Sun'], t0, t1, horizon)
    # t = ts.linspace(t0, t1, 5)
    t -= dt.timedelta(hours=7)
    #strings = t.utc_strftime('%m-%d %H%M:%S.%f')
    #strings = t.utc_strftime('%m-%d %H%M')
    #print(strings)

    skyfield_rises = []

    for ti in t:
        u = ti.utc
        tup = u.month, u.day, u.hour * 60 + u.minute + (u.second >= 30)
        skyfield_rises.append(tup)

    for u, s in zip(usno_rises, skyfield_rises):
        #print(u[0:2], s[0:2])
        if u[0:2] != s[0:2]:
            print()
            print('Error: usno', u[0:2], 'skyfield', s[0:2])
            exit(1)
        print(u[2] - s[2], end=' ')
    print()

    ti = t[2]
    ti += dt.timedelta(hours=7)
    #print(ti.utc_strftime())
    alt, az, _ = observer.at(ti).observe(e['Sun']).apparent().altaz()
    print('Altitude:', alt.degrees)
    #01  1347 0249
    #02  1420 0350
    #03  1457 0451
    #04  1540

if __name__ == '__main__':
    main()

