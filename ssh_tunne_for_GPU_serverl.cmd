@echo off
title SSH Tunnel to VDS (Port 9001)
echo runing SSH-tunnel...

:loop
rem ssh -v -N -R 9001:127.0.0.1:9001 yrsolo@89.169.132.198 -o ServerAliveInterval=30

ssh -v ^
    -N ^
    -R 9001:127.0.0.1:9001 ^
    -o ExitOnForwardFailure=yes ^
    -o ServerAliveInterval=60 ^
    -o ServerAliveCountMax=3 ^
    yrsolo@89.169.132.198


echo ERROR. Reboot after 5 sec...
timeout /t 5
goto loop