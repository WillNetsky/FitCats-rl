#!/bin/bash

echo "How many agents do you want to launch?"
read COUNT

for (( i=1; i<=COUNT; i++ ))
do
   # Launch each instance in a new terminal tab/window
   # This depends on your terminal emulator. Assuming gnome-terminal.
   gnome-terminal --tab --title="Agent $i" -- bash -c "./launch_instance.sh $i; exec bash"

   echo "Launched Agent $i"
   sleep 2 # Stagger launches slightly
done

echo "All agents launched!"
