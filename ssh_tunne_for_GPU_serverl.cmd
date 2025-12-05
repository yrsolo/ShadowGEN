@echo off
set SERVER=89.169.132.198
set USER=yrsolo
set RPORT=9001
set LPORT=9001

:loop
echo.
echo ==== CLEAN OLD TUNNELS ON %SERVER%:%RPORT% ====

ssh %USER%@%SERVER% "echo '=== BEFORE ==='; sudo ss -tulpn | grep %RPORT% || echo 'no listeners'; pids=$(sudo lsof -ti:%RPORT% -sTCP:LISTEN 2>/dev/null); echo PIDS=$pids; if [ -n \"$pids\" ]; then echo KILLING $pids; sudo kill $pids; fi; echo '=== AFTER ==='; sudo ss -tulpn | grep %RPORT% || echo 'no listeners'"

echo ==== START SSH TUNNEL %RPORT% - 127.0.0.1:%LPORT% ====

ssh -v ^
  -N ^
  -R %RPORT%:127.0.0.1:%LPORT% ^
  -o ExitOnForwardFailure=yes ^
  -o ServerAliveInterval=60 ^
  -o ServerAliveCountMax=3 ^
  %USER%@%SERVER%

echo ERROR: SSH exited, restart in 5 sec...
timeout /t 5 >nul
goto loop
