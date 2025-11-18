def _on_swirl(self, msg: Bool):
    """
    Track when RoboHand is in SWIRL phase and control pump accordingly.

    Behavior:
    - When swirl_active becomes True and the pump hasn't been started for this swirl,
      start the pump for swirl_time_s seconds (parameter).
    - When swirl_active becomes False, stop the pump if it was started, and reset state.
    """
    new_state = bool(msg.data)
    self.swirl_active = new_state

    # read configured swirl time (fallback to 8.0s if param missing)
    try:
        swirl_time = float(self.get_parameter('swirl_time_s').value)
    except Exception:
        swirl_time = 8.0

    # Guard flag to avoid repeatedly commanding the pump while swirl_active remains True
    if new_state and not getattr(self, '_did_start_pump', False):
        # start pump for swirl_time seconds at full duty (adjust duty if desired)
        duty = 1.0
        self._pump_on(duty=duty, duration_s=swirl_time)
        self._did_start_pump = True

    # If swirl ended and pump was started, stop it and reset flag
    if not new_state and getattr(self, '_did_start_pump', False):
        self._pump_off()
        self._did_start_pump = False
