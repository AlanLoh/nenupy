#! /usr/bin/python3.6
# -*- coding: utf-8 -*-

"""
Class to do fancy progressbars
        by G. Schworer
https://github.com/ceyzeriat/patiencebar
"""

from sys import stdout as _stdout

class ProgressBar(object):
    """
    Provides a terminal-friendly single-thread progress bar
    Args:
      * valmax (float): the finish value of the progress bar. Default is 100.
      * barsize (int >0): the size of the bar in the opened terminal. If ``None``, the bar will automatically fit the width of the window.
      * title (str): the title, printed one line above the progress bar
      * bar (bool): whether the bar should be displayed or not. If ``False``, only the text given at each :func:`update` will be printed
      * up_every (int [0-100]): if ``bar`` is ``True``, the progress bar will be updated every ``up_every`` percent of progress. Setting ``up_every`` = 0 updates the progress bar at each :func:`update`
    >>> import patiencebar as PB
    >>> n_calc = 34
    >>> pb = PB.ProgressBar(valmax=n_calc, barsize=50, title="Test bar")
    >>> for i in range(n_calc):
    >>>     do_stuff()
    >>>     pb.update()
    """
    def __init__(self, valmax=100, barsize=None, title=None, bar=True, up_every=2):
        self.reset(valmax=valmax, barsize=barsize, title=title, bar=bar, up_every=up_every)

    @property
    def valmax(self):
        return self._valmax
    @valmax.setter
    def valmax(self, value):
        raise AttributeError("Read-only")

    @property
    def barsize(self):
        return self._barsize
    @barsize.setter
    def barsize(self, value):
        raise AttributeError("Read-only")

    @property
    def title(self):
        return self._title
    @title.setter
    def title(self, value):
        raise AttributeError("Read-only")

    @property
    def running(self):
        return self._running
    @running.setter
    def running(self, value):
        raise AttributeError("Read-only")

    @property
    def bar(self):
        return self._bar
    @bar.setter
    def bar(self, value):
        raise AttributeError("Read-only")

    @property
    def up_every(self):
        return self._up_every
    @up_every.setter
    def up_every(self, value):
        raise AttributeError("Read-only")


    def reset(self, valmax=None, barsize=None, title=None, bar=None, up_every=None):
        """
        Resets the progress bar with initialization values, unless new values are given
        Args:
          * valmax (float): the finish value of the progress bar. Default is 100.
          * barsize (int >0): the size of the bar in the opened terminal. If ``None``, the bar will automatically fit the width of the window.
          * title (str): the title, printed one line above the progress bar.
          * bar (bool): whether the bar should be displayed or not. If ``False``, only the text given at each :func:`update` will be printed.
          * up_every (int [0-100]): if ``bar`` is ``True``, the progress bar will be updated every ``up_every`` percent of progress. Setting ``up_every`` = 0 updates the progress bar at each :func:`update`.
        >>> n_calc = 34
        >>> pb = PB.Patiencebar(valmax=n_calc, barsize=50, title="Test bar")
        >>> for i in range(n_calc):
        >>>     do_stuff()
        >>>     pb.update()
        >>> pb.reset(title="Second trial", barsize=70)
        >>> for i in range(n_calc):
        >>>     do_stuff()
        >>>     pb.update()
        """
        self.win_width = 50 #_get_terminal_size()[1]
        self.step = 0
        self._nextup = 0
        self._title_written = False
        self._running = False
        if valmax is not None:
            self._valmax = float(valmax)
        else:
            self._valmax = float(getattr(self, 'valmax', 100))
        if barsize is not None:
            self._barsize = int(barsize)
        else:
            self._barsize = int(getattr(self, 'barsize', self.win_width-8))
        if title is not None:
            self._title = str(title)
        elif getattr(self, 'title', None) is not None:
            self._title = str(getattr(self, 'title'))
        else:
            self._title = None
        if bar is not None:
            self._bar = bool(bar)
        else:
            self._bar = bool(getattr(self, 'bar', True))
        if up_every is not None:
            self._up_every = int(min(100,max(0,up_every)))
        else:
            self._up_every = int(getattr(self, 'up_every', 2))

    
    def update(self, step=None):
        """
        Updates the progress bar to a newer value
        Args:
          * step (None): adds 1 to the progress of the bar, where ``valmax`` is the finish value.
          * step (float): sets the progress of the bar to the ``step`` value, where ``valmax`` is the finish value.
          * step (str): displays ``step`` on a new line. For this to work, ``bar`` must be ``False`` (no progress bar displayed) otherwise the update instruction is ignored.
        """
        self._doupdate(step=step)
        
    def _doupdate(self, step=None):
        # initialization
        if not self._title_written:
            if self.title is not None: print(self.title)
            self._title_written = True
            self._running = True # in case it was not yet done
        if self.bar: # if display progressbar
            if step is None: # no step given
                self.step += 1
            else: # step given              
                try: # make sure we got some number
                    self.step = min(float(step), self.valmax)
                except: # if not.. WTF.. don't change anything
                    return
            # render bar
            perc = int(self.step*100./self.valmax)
            if perc < self._nextup: return
            self._nextup = min(100, perc+self._up_every)
            bar = int(perc/100.*self.barsize)
            spacing = (self.win_width-self.barsize-8)/2.
            out = '\r%s[%s%s] %3d %%%s' % (' ' * int(spacing), '=' * bar, ' ' * int(self.barsize - bar), perc, ' ' * int(spacing+0.5))
            _stdout.write(out)
            _stdout.flush()
        else: # if no bar display
            if step is None: step = 'tic'
            print(str(step))
            self.step += 1
        if self.step >= self.valmax: # finished
            print("\r")
            self._running = False            