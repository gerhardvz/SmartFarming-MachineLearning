import unittest
import Plant_Identifier as PI
from Plant_Identifier import ResNet9

def main():
    import time

    start_time = time.time()

    PI.init()
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    PI.test("test/Test.JPG")
    PI.test( "test/DiseaseFigure86.jpg")
    PI.test( "test/GrapeHealthy.jpg")
    PI.test( "test/CornCommonRust2.JPG")

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
