from html import escape

def build_alert_banner(state: str = "standby", source: str = "image") -> str:
    return ""

def build_alert_controls() -> str:
    return ""

def build_alert_signal(alert_active: bool, status: str, source: str) -> str:
    import uuid
    alert_id = str(uuid.uuid4())
    title = "Accident Detected" if alert_active else "Alert Standby"
    message = "The uploaded media was classified as an accident." if alert_active else ""
    return f"""
    <div
        class="alert-signal-data"
        data-alert-id="{alert_id}"
        data-alert-active="{str(alert_active).lower()}"
        data-alert-status="{escape(status)}"
        data-alert-source="{escape(source)}"
        data-alert-title="{escape(title)}"
        data-alert-message="{escape(message)}"
    ></div>
    """

def build_alert_controller_head() -> str:
    return """
    <script>
    (() => {
      if (window.__accidentAlertControllerLoaded) return;
      window.__accidentAlertControllerLoaded = true;

      const controller = {
        state: {
          id: "",
          active: false,
          status: "idle",
          source: "image",
          title: "Accident Detected",
          message: "The uploaded media was classified as an accident.",
        },
        audioSupported: Boolean(window.AudioContext || window.webkitAudioContext),
        notificationSupported: typeof window.Notification !== "undefined",
        soundEnabled: true,
        hasUserInteracted: false,
        audioContext: null,
        alarmTimer: null,
        alarmTimeout: null,
        signalObserver: null,
        signalObserverHost: null,
        attachTimer: null,
        handlersBound: false,
        lastNotificationKey: "",

        init() {
          this.bindGlobalHandlers();
          this.attachSignalObserver();
          this.syncFromSignal();
        },

        enableSoundAndNotifications() {
          this.soundEnabled = true;
          this.registerInteraction();
          this.ensureAudioContext();

          if (this.notificationSupported && Notification.permission === "default") {
            Notification.requestPermission()
              .then((permission) => {
                if (permission === "granted") {
                  this.lastNotificationKey = "";
                  this.notify();
                }
              })
              .catch(() => {});
          }
        },

        bindGlobalHandlers() {
          if (this.handlersBound) return;
          this.handlersBound = true;

          document.addEventListener("pointerdown", () => {
            this.registerInteraction();
          }, true);

          document.addEventListener("keydown", () => {
            this.registerInteraction();
          }, true);

          document.addEventListener("click", (event) => {
            const button = event.target.closest("button");
            if (!button) return;

            if (button.id === "run-image-inference-btn" || button.id === "run-video-inference-btn") {
              this.enableSoundAndNotifications();
            }
          }, true);

          window.addEventListener("beforeunload", () => this.destroy(), { once: true });
          window.addEventListener("pagehide", () => this.destroy(), { once: true });
        },

        registerInteraction() {
          this.hasUserInteracted = true;
          if (this.soundEnabled) {
            this.ensureAudioContext();
          }
        },

        ensureAudioContext() {
          if (!this.audioSupported) return null;
          const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
          if (!AudioContextCtor) return null;

          try {
            if (!this.audioContext) {
              this.audioContext = new AudioContextCtor();
            }
            if (this.audioContext.state === "suspended") {
              this.audioContext.resume().catch(() => {});
            }
            return this.audioContext;
          } catch (error) {
            return null;
          }
        },

        attachSignalObserver() {
          const host = document.getElementById("alert-signal-region");
          if (!host) {
            window.clearTimeout(this.attachTimer);
            this.attachTimer = window.setTimeout(() => this.attachSignalObserver(), 250);
            return;
          }

          if (this.signalObserverHost === host) return;

          if (this.signalObserver) {
            this.signalObserver.disconnect();
          }

          this.signalObserverHost = host;
          this.signalObserver = new MutationObserver(() => this.syncFromSignal());
          this.signalObserver.observe(host, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
          });
        },

        readSignalPayload() {
          const host = document.getElementById("alert-signal-region");
          const node = host ? host.querySelector(".alert-signal-data") : null;
          if (!node) {
            return {
              id: "",
              active: false,
              status: "idle",
              source: "image",
              title: "Accident Detected",
              message: "The uploaded media was classified as an accident.",
            };
          }

          return {
            id: node.dataset.alertId || "",
            active: node.dataset.alertActive === "true",
            status: node.dataset.alertStatus || "idle",
            source: node.dataset.alertSource || "image",
            title: node.dataset.alertTitle || "Accident Detected",
            message: node.dataset.alertMessage || "The uploaded media was classified as an accident.",
          };
        },

        syncFromSignal() {
          const next = this.readSignalPayload();
          const wasActive = this.state.active;
          const previousStatus = this.state.status;
          const previousId = this.state.id;
          this.state = next;

          if (!next.active) {
            if (wasActive || previousStatus !== next.status || previousId !== next.id) {
              this.resetRuntime();
            } else {
              this.applyClasses();
            }
            return;
          }

          if (!wasActive || previousId !== next.id) {
            this.stopAlarm();
            this.activateAlert();
            return;
          }

          this.applyClasses();
        },

        activateAlert() {
          this.applyClasses();
          this.triggerVibration();

          if (this.soundEnabled && this.hasUserInteracted) {
            this.startAlarm();
          }

          this.notify();
        },

        resetRuntime() {
          this.stopAlarm();
          this.stopVibration();
          this.lastNotificationKey = "";
          this.applyClasses();
        },

        startAlarm() {
          if (!this.state.active || !this.soundEnabled || !this.hasUserInteracted) return;
          if (!this.ensureAudioContext()) return;
          if (this.alarmTimer) return;

          this.playAlarmPattern();
          this.alarmTimer = window.setInterval(() => {
            this.playAlarmPattern();
          }, 1900);

          if (this.alarmTimeout) {
            window.clearTimeout(this.alarmTimeout);
          }
          this.alarmTimeout = window.setTimeout(() => {
            this.stopAlarm();
          }, 10000);
        },

        stopAlarm() {
          if (this.alarmTimer) {
            window.clearInterval(this.alarmTimer);
            this.alarmTimer = null;
          }
          if (this.alarmTimeout) {
            window.clearTimeout(this.alarmTimeout);
            this.alarmTimeout = null;
          }
        },

        playAlarmPattern() {
          const ctx = this.ensureAudioContext();
          if (!ctx) return;

          const pulses = [
            { at: 0.0, freq: 740, duration: 0.16 },
            { at: 0.26, freq: 620, duration: 0.16 },
            { at: 0.56, freq: 760, duration: 0.24 },
          ];

          pulses.forEach((pulse) => {
            try {
              const oscillator = ctx.createOscillator();
              const gainNode = ctx.createGain();
              const startAt = ctx.currentTime + pulse.at;
              const stopAt = startAt + pulse.duration;

              oscillator.type = "square";
              oscillator.frequency.setValueAtTime(pulse.freq, startAt);
              gainNode.gain.setValueAtTime(0.0001, startAt);
              gainNode.gain.exponentialRampToValueAtTime(0.06, startAt + 0.02);
              gainNode.gain.exponentialRampToValueAtTime(0.0001, stopAt);

              oscillator.connect(gainNode);
              gainNode.connect(ctx.destination);
              oscillator.start(startAt);
              oscillator.stop(stopAt + 0.03);
              oscillator.onended = () => {
                try {
                  oscillator.disconnect();
                  gainNode.disconnect();
                } catch (error) {
                }
              };
            } catch (error) {
            }
          });
        },

        triggerVibration() {
          if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") {
            try {
              navigator.vibrate([300, 150, 300, 150, 500]);
            } catch (error) {
            }
          }
        },

        stopVibration() {
          if (typeof navigator !== "undefined" && typeof navigator.vibrate === "function") {
            try {
              navigator.vibrate(0);
            } catch (error) {
            }
          }
        },

        notify() {
          if (!this.notificationSupported) return;
          if (Notification.permission !== "granted") return;

          const notificationKey = `${this.state.source}:${this.state.status}`;
          if (this.lastNotificationKey === notificationKey) return;
          this.lastNotificationKey = notificationKey;

          try {
            new Notification(this.state.title || "Accident Detected", {
              body: this.state.message || "The uploaded media was classified as an accident.",
              tag: "capstone-accident-alert",
              renotify: true,
            });
          } catch (error) {
          }
        },

        enableBrowserAlerts() {
          this.registerInteraction();
          if (!this.notificationSupported) {
            return;
          }

          if (Notification.permission === "granted") {
            this.notify();
            return;
          }

          Notification.requestPermission()
            .then((permission) => {
              if (permission === "granted") {
                this.lastNotificationKey = "";
                this.notify();
              }
            })
            .catch(() => {});
        },

        applyClasses() {
          const body = document.body;
          if (!body) return;

          body.classList.toggle("accident-alert-active", this.state.active);
          body.classList.toggle("accident-alert-source-image", this.state.active && this.state.source === "image");
          body.classList.toggle("accident-alert-source-video", this.state.active && this.state.source === "video");
        },

        setPill(id, text, tone) {
          const pill = document.getElementById(id);
          if (!pill) return;
          pill.textContent = text;
          pill.className = `alert-pill ${tone}`;
        },

        destroy() {
          this.stopAlarm();
          this.stopVibration();
          if (this.signalObserver) {
            this.signalObserver.disconnect();
          }
          window.clearTimeout(this.attachTimer);
          try {
            if (this.audioContext && typeof this.audioContext.close === "function" && this.audioContext.state !== "closed") {
              this.audioContext.close().catch(() => {});
            }
          } catch (error) {
            console.warn("AudioContext close failed:", error);
          }
        },
      };

      window.__accidentAlertController = controller;

      const boot = () => controller.init();
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
      } else {
        boot();
      }
    })();
    </script>
    """
