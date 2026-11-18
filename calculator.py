from math import sin, cos, atan, sqrt, radians, log

# Physics model matches Create Big Cannons 5.11.7 (mc 1.21.1):
#   AbstractCannonProjectile.tick():
#       acceleration = -drag * velocity + (0, gravity, 0)
#       position    += velocity + 0.5 * acceleration
#       velocity    += acceleration
#   Muzzle velocity = total charge power (powder charge = 2 per block, big cartridge = 2 per power level)
#   Projectile spawns at the centre of the last barrel block (the muzzle).
#
# Original brute-force idea and formulas: @sashafiesta on Discord, calculator by Malex21.

DEFAULT_GRAVITY = -0.05
DEFAULT_DRAG = 0.01
POWER_PER_CHARGE = 2.0

MINIMUM_PITCH = -30.0
MAXIMUM_PITCH = 60.0
SCAN_STEP_DEGREES = 0.25

# Values from data/minecraft/dimension_munition_properties in the CBC jar
DIMENSION_MULTIPLIERS = {
    "overworld": (1.0, 1.0),   # (gravity multiplier, drag multiplier)
    "nether": (1.1, 1.1),
    "end": (0.8, 0.0),
}

# Per-projectile gravity / drag / added charge power from munition_properties/projectiles
PROJECTILE_PROPERTIES = {
    "default": (-0.05, 0.01, 0.0),       # shot, HE, AP, shrapnel, smoke, fluid shells
    "mortar_stone": (-0.025, 0.01, 0.0),
}


class OutOfRangeException(Exception):
    pass


class BallisticModel:
    """
    Closed-form version of CBC's per-tick projectile update.
    Works with fractional tick counts so the solver can root-find smoothly.
    """

    def __init__(self, gravity, drag):
        self.gravity = gravity
        self.drag = drag
        self.positionFactor = 1.0 - drag / 2.0
        self.velocityRetention = 1.0 - drag

    def velocitySum(self, ticks):
        """Sum of retention^k for k = 0 .. ticks-1."""
        if self.drag == 0:
            return ticks
        return (1.0 - self.velocityRetention ** ticks) / self.drag

    def horizontalDistance(self, horizontalSpeed, ticks):
        return self.positionFactor * horizontalSpeed * self.velocitySum(ticks)

    def verticalDistance(self, verticalSpeed, ticks):
        if self.drag == 0:
            return verticalSpeed * ticks + self.gravity * ticks * ticks / 2.0
        velocitySum = self.velocitySum(ticks)
        gravityPart = (self.gravity / self.drag) * (ticks - velocitySum)
        return self.positionFactor * (verticalSpeed * velocitySum + gravityPart) + self.gravity * ticks / 2.0

    def ticksToCover(self, horizontalSpeed, horizontalDistance):
        """Inverse of horizontalDistance. Returns None if drag stops the shell before it gets there."""
        if horizontalSpeed <= 0:
            return None
        if self.drag == 0:
            return horizontalDistance / (self.positionFactor * horizontalSpeed)
        remainingFraction = 1.0 - horizontalDistance * self.drag / (self.positionFactor * horizontalSpeed)
        if remainingFraction <= 0:
            return None
        return log(remainingFraction) / log(self.velocityRetention)


def simulateClosestApproach(model, muzzleSpeed, pitchDegrees, muzzleOffset, horizontalDistance, heightDifference, maximumTicks=2000):
    """
    Tick-by-tick simulation exactly like the mod, used to measure how close the shell actually gets.
    Returns the smallest distance (in blocks) between the shell's path and the target.
    """
    pitchRadians = radians(pitchDegrees)
    horizontalPosition = muzzleOffset * cos(pitchRadians)
    verticalPosition = muzzleOffset * sin(pitchRadians)
    horizontalVelocity = muzzleSpeed * cos(pitchRadians)
    verticalVelocity = muzzleSpeed * sin(pitchRadians)

    closestDistance = float("inf")
    for _ in range(maximumTicks):
        horizontalAcceleration = -model.drag * horizontalVelocity
        verticalAcceleration = -model.drag * verticalVelocity + model.gravity

        nextHorizontal = horizontalPosition + horizontalVelocity + 0.5 * horizontalAcceleration
        nextVertical = verticalPosition + verticalVelocity + 0.5 * verticalAcceleration

        # Closest point on this tick's straight segment (CBC clips along the segment too)
        segmentHorizontal = nextHorizontal - horizontalPosition
        segmentVertical = nextVertical - verticalPosition
        segmentLengthSquared = segmentHorizontal ** 2 + segmentVertical ** 2
        if segmentLengthSquared > 0:
            segmentFraction = ((horizontalDistance - horizontalPosition) * segmentHorizontal + (heightDifference - verticalPosition) * segmentVertical) / segmentLengthSquared
            segmentFraction = max(0.0, min(1.0, segmentFraction))
            closestHorizontal = horizontalPosition + segmentHorizontal * segmentFraction
            closestVertical = verticalPosition + segmentVertical * segmentFraction
            closestDistance = min(closestDistance, sqrt((closestHorizontal - horizontalDistance) ** 2 + (closestVertical - heightDifference) ** 2))

        horizontalPosition, verticalPosition = nextHorizontal, nextVertical
        horizontalVelocity += horizontalAcceleration
        verticalVelocity += verticalAcceleration

        if verticalPosition < heightDifference - 64 and verticalVelocity < 0:
            break
        if horizontalPosition > horizontalDistance + 64:
            break
    return closestDistance

def simulateLandingDistance(model, muzzleSpeed, pitchDegrees, muzzleOffset, heightDifference, maximumTicks=4000):
    """
    Tick-by-tick simulation: horizontal distance at which the shell comes back down to heightDifference.
    Returns None if it never gets there.
    """
    pitchRadians = radians(pitchDegrees)
    horizontalPosition = muzzleOffset * cos(pitchRadians)
    verticalPosition = muzzleOffset * sin(pitchRadians)
    horizontalVelocity = muzzleSpeed * cos(pitchRadians)
    verticalVelocity = muzzleSpeed * sin(pitchRadians)

    for _ in range(maximumTicks):
        horizontalAcceleration = -model.drag * horizontalVelocity
        verticalAcceleration = -model.drag * verticalVelocity + model.gravity
        nextHorizontal = horizontalPosition + horizontalVelocity + 0.5 * horizontalAcceleration
        nextVertical = verticalPosition + verticalVelocity + 0.5 * verticalAcceleration

        if verticalVelocity < 0 and nextVertical <= heightDifference <= verticalPosition:
            crossingFraction = (verticalPosition - heightDifference) / (verticalPosition - nextVertical)
            return horizontalPosition + (nextHorizontal - horizontalPosition) * crossingFraction

        horizontalPosition, verticalPosition = nextHorizontal, nextVertical
        horizontalVelocity += horizontalAcceleration
        verticalVelocity += verticalAcceleration
    return None


def CalibratePowerPerCharge(cannon, landing, power, pitch, lenght, dimension="overworld", projectile="default"):
    """Works out the real power per charge from one test shot.

    Args:
        cannon (tuple): Position of the cannon block held by the mount (x, y, z)
        landing (tuple): Where the test shell actually hit (x, y, z)
        power (int): Number of charges used for the test shot
        pitch (float): Pitch the cannon actually had when firing (read it with goggles)
        lenght (int): Blocks from the mount-held block to the muzzle, both included

    Returns:
        float: power per charge to type into the calculator (vanilla CBC = 2)
    """
    projectileGravity, projectileDrag, addedChargePower = PROJECTILE_PROPERTIES.get(projectile, PROJECTILE_PROPERTIES["default"])
    gravityMultiplier, dragMultiplier = DIMENSION_MULTIPLIERS.get(dimension, DIMENSION_MULTIPLIERS["overworld"])
    model = BallisticModel(projectileGravity * gravityMultiplier, projectileDrag * dragMultiplier)

    muzzleOffset = max(lenght - 1, 0)
    landingDistance = sqrt((landing[0] - cannon[0]) ** 2 + (landing[2] - cannon[2]) ** 2)
    heightDifference = landing[1] - cannon[1]

    lowestSpeed, highestSpeed = 0.1, 100.0
    if (simulateLandingDistance(model, highestSpeed, pitch, muzzleOffset, heightDifference) or 0) < landingDistance:
        raise OutOfRangeException("Couldn't match that test shot - check the landing coordinates and pitch.")

    for _ in range(80):
        middleSpeed = (lowestSpeed + highestSpeed) / 2
        reachedDistance = simulateLandingDistance(model, middleSpeed, pitch, muzzleOffset, heightDifference)
        if reachedDistance is None or reachedDistance < landingDistance:
            lowestSpeed = middleSpeed
        else:
            highestSpeed = middleSpeed

    muzzleSpeed = (lowestSpeed + highestSpeed) / 2
    return round((muzzleSpeed - addedChargePower) / power, 3)

def BallisticsToTarget(cannon, target, power, direction, lenght, dimension="overworld", projectile="default", powerPerCharge=POWER_PER_CHARGE):
    """Calculates yaw and the two pitch solutions (shallow and steep) to hit a target.

    Args:
        cannon (tuple): Position of the cannon block held by the mount (x, y, z)
        target (tuple): Position of the target block (x, y, z)
        power (int): Number of powder charges (big cartridge: total power levels)
        direction (str): Direction the cannon faces when not mounted (north/south/east/west)
        lenght (int): Blocks from the mount-held block to the muzzle, both included
        dimension (str): overworld / nether / end
        projectile (str): key of PROJECTILE_PROPERTIES

    Returns:
        tuple of two tuples: (yaw, pitch, airtimeTicks, airtimeSeconds, fuzeTime, precision)
    """

    projectileGravity, projectileDrag, addedChargePower = PROJECTILE_PROPERTIES.get(projectile, PROJECTILE_PROPERTIES["default"])
    gravityMultiplier, dragMultiplier = DIMENSION_MULTIPLIERS.get(dimension, DIMENSION_MULTIPLIERS["overworld"])
    model = BallisticModel(projectileGravity * gravityMultiplier, projectileDrag * dragMultiplier)

    muzzleSpeed = power * powerPerCharge + addedChargePower
    if muzzleSpeed <= 0:
        raise OutOfRangeException("You need at least one charge !")

    # Projectile spawns at the centre of the muzzle block, which is (lenght - 1) blocks from the mount-held block's centre
    muzzleOffset = max(lenght - 1, 0)

    deltaX, deltaZ = (cannon[0] - target[0], cannon[2] - target[2])
    horizontalDistance = sqrt(deltaX * deltaX + deltaZ * deltaZ)
    heightDifference = target[1] - cannon[1]

    # --- Yaw (unchanged from the original calculator) ---
    if deltaX != 0:
        yaw = atan(deltaZ / deltaX) * 57.2957795131
    else:
        yaw = 90
    if deltaX >= 0:
        yaw += 180

    if direction == "north":
        yaw = (yaw + 90) % 360
    elif direction == "west":
        yaw = (yaw + 180) % 360
    elif direction == "south":
        yaw = (yaw + 270) % 360
    elif direction != "east":
        return "Invalid direction"

    # --- Pitch: find angles where the shell passes exactly through the target ---
    def heightError(pitchDegrees):
        """Shell height minus target height at the moment it reaches the target horizontally. None if it never gets there."""
        pitchRadians = radians(pitchDegrees)
        remainingDistance = horizontalDistance - muzzleOffset * cos(pitchRadians)
        if remainingDistance < 0:
            return None
        ticks = model.ticksToCover(muzzleSpeed * cos(pitchRadians), remainingDistance)
        if ticks is None:
            return None
        shellHeight = muzzleOffset * sin(pitchRadians) + model.verticalDistance(muzzleSpeed * sin(pitchRadians), ticks)
        return shellHeight - heightDifference

    def airtimeAt(pitchDegrees):
        pitchRadians = radians(pitchDegrees)
        return model.ticksToCover(muzzleSpeed * cos(pitchRadians), horizontalDistance - muzzleOffset * cos(pitchRadians))

    scannedPitches = []
    currentPitch = MINIMUM_PITCH
    while currentPitch <= 90.0:
        scannedPitches.append((currentPitch, heightError(currentPitch)))
        currentPitch += SCAN_STEP_DEGREES

    solutions = []
    for (lowerPitch, lowerError), (upperPitch, upperError) in zip(scannedPitches, scannedPitches[1:]):
        if lowerError is None or upperError is None:
            continue
        if lowerError == 0:
            solutions.append(lowerPitch)
            continue
        if (lowerError < 0) != (upperError < 0):
            # Bisection between the two scanned angles
            for _ in range(60):
                middlePitch = (lowerPitch + upperPitch) / 2
                middleError = heightError(middlePitch)
                if middleError is None:
                    break
                if (middleError < 0) == (lowerError < 0):
                    lowerPitch, lowerError = middlePitch, middleError
                else:
                    upperPitch = middlePitch
            solutions.append((lowerPitch + upperPitch) / 2)

    if not solutions:
        raise OutOfRangeException("The target is unreachable with your current cannon configuration !")

    shallowPitch = solutions[0]
    steepPitch = solutions[-1]

    def buildResult(pitchDegrees):
        airtimeTicks = airtimeAt(pitchDegrees)
        missDistance = simulateClosestApproach(model, muzzleSpeed, round(pitchDegrees, 2), muzzleOffset, horizontalDistance, heightDifference)
        precision = round(max(0.0, 1 - missDistance / max(horizontalDistance, 1.0)) * 100, 2)

        # Timed fuze counts down once per projectile tick; same "10 ticks early" offset as the original version
        fuzeTime = max(1, int(airtimeTicks) - 10)

        if pitchDegrees > MAXIMUM_PITCH + 0.5:
            shownPitch = "Over 60"
        elif pitchDegrees < MINIMUM_PITCH - 0.5:
            shownPitch = "Under -30"
        else:
            shownPitch = round(pitchDegrees, 2)

        return (
            round(yaw, 2) + 0.0,
            shownPitch,
            round(airtimeTicks, 1),
            round(airtimeTicks / 20, 2),
            fuzeTime,
            precision,
        )

    return buildResult(shallowPitch), buildResult(steepPitch)


# CREDITS OF ORIGINAL FORMULAS : @sashafiesta#1978 on Discord