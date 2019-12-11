import RPi.GPIO as GPIO
import smbus
import time
#0x48

address = 0x48
A0 = 0x40
A1 = 0x41
A2 = 0x42
A3 = 0x43

bus = smbus.SMBus(1)
try:
    while True:
        #A3:variable v
        #A0:input v
        bus.write_byte(address,A0)#bus.write_byte(address,A0)
        #value = 143 - bus.read_byte(address)
        depth = bus.read_byte(address)/256*5/3*100
        print('depth:','%.3f'%depth,'cm')
        #print('%#x'%value)
        time.sleep(1)

except KeyboardInterrupt:
    print('Stoped by user!')
    pass

finally:
    #GPIO.cleanup()
    print('Clean GPIO set!')
