from dataclasses import dataclass

@dataclass(frozen=True)
class SpaceDefinition:
    index: int
    name: str
    space_type: str                # "property", "railroad", "utility", "tax", "chance", "community_chest", "go", "jail", "free_parking", "go_to_jail"
    group: int = 0                 # 0=non-property, 1=railroad, 2=utility, 3-10=color groups
    price: int = 0
    base_rent: int = 0
    rent_1h: int = 0
    rent_2h: int = 0
    rent_3h: int = 0
    rent_4h: int = 0
    rent_hotel: int = 0
    house_price: int = 0
    mortgage_value: int = 0
    tax_amount: int = 0
    color_name: str = ""

COLOR_NAMES = {
    3: "Brown",
    4: "Light Blue",
    5: "Pink",
    6: "Orange",
    7: "Red",
    8: "Yellow",
    9: "Green",
    10: "Dark Blue",
}

# Positions of special spaces
GO_POSITION = 0
JAIL_POSITION = 10
FREE_PARKING_POSITION = 20
GO_TO_JAIL_POSITION = 30

BOARD: list[SpaceDefinition] = [
    SpaceDefinition(0, "GO", "go"),
    SpaceDefinition(1, "Mediterranean Avenue", "property", 3, 60, 2, 10, 30, 90, 160, 250, 50, 30, 0, "Brown"),
    SpaceDefinition(2, "Community Chest", "community_chest"),
    SpaceDefinition(3, "Baltic Avenue", "property", 3, 60, 4, 20, 60, 180, 320, 450, 50, 30, 0, "Brown"),
    SpaceDefinition(4, "Income Tax", "tax", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200),
    SpaceDefinition(5, "Reading Railroad", "railroad", 1, 200, 0, 0, 0, 0, 0, 0, 0, 100),
    SpaceDefinition(6, "Oriental Avenue", "property", 4, 100, 6, 30, 90, 270, 400, 550, 50, 50, 0, "Light Blue"),
    SpaceDefinition(7, "Chance", "chance"),
    SpaceDefinition(8, "Vermont Avenue", "property", 4, 100, 6, 30, 90, 270, 400, 550, 50, 50, 0, "Light Blue"),
    SpaceDefinition(9, "Connecticut Avenue", "property", 4, 120, 8, 40, 100, 300, 450, 600, 50, 60, 0, "Light Blue"),
    SpaceDefinition(10, "Just Visiting / Jail", "jail"),
    SpaceDefinition(11, "St. Charles Place", "property", 5, 140, 10, 50, 150, 450, 625, 750, 100, 70, 0, "Pink"),
    SpaceDefinition(12, "Electric Company", "utility", 2, 150, 0, 0, 0, 0, 0, 0, 0, 75),
    SpaceDefinition(13, "States Avenue", "property", 5, 140, 10, 50, 150, 450, 625, 750, 100, 70, 0, "Pink"),
    SpaceDefinition(14, "Virginia Avenue", "property", 5, 160, 12, 60, 180, 500, 700, 900, 100, 80, 0, "Pink"),
    SpaceDefinition(15, "Pennsylvania Railroad", "railroad", 1, 200, 0, 0, 0, 0, 0, 0, 0, 100),
    SpaceDefinition(16, "St. James Place", "property", 6, 180, 14, 70, 200, 550, 750, 950, 100, 90, 0, "Orange"),
    SpaceDefinition(17, "Community Chest", "community_chest"),
    SpaceDefinition(18, "Tennessee Avenue", "property", 6, 180, 14, 70, 200, 550, 750, 950, 100, 90, 0, "Orange"),
    SpaceDefinition(19, "New York Avenue", "property", 6, 200, 16, 80, 220, 600, 800, 1000, 100, 100, 0, "Orange"),
    SpaceDefinition(20, "Free Parking", "free_parking"),
    SpaceDefinition(21, "Kentucky Avenue", "property", 7, 220, 18, 90, 250, 700, 875, 1050, 150, 110, 0, "Red"),
    SpaceDefinition(22, "Chance", "chance"),
    SpaceDefinition(23, "Indiana Avenue", "property", 7, 220, 18, 90, 250, 700, 875, 1050, 150, 110, 0, "Red"),
    SpaceDefinition(24, "Illinois Avenue", "property", 7, 240, 20, 100, 300, 750, 925, 1100, 150, 120, 0, "Red"),
    SpaceDefinition(25, "B&O Railroad", "railroad", 1, 200, 0, 0, 0, 0, 0, 0, 0, 100),
    SpaceDefinition(26, "Atlantic Avenue", "property", 8, 260, 22, 110, 330, 800, 975, 1150, 150, 130, 0, "Yellow"),
    SpaceDefinition(27, "Ventnor Avenue", "property", 8, 260, 22, 110, 330, 800, 975, 1150, 150, 130, 0, "Yellow"),
    SpaceDefinition(28, "Water Works", "utility", 2, 150, 0, 0, 0, 0, 0, 0, 0, 75),
    SpaceDefinition(29, "Marvin Gardens", "property", 8, 280, 24, 120, 360, 850, 1025, 1200, 150, 140, 0, "Yellow"),
    SpaceDefinition(30, "Go to Jail", "go_to_jail"),
    SpaceDefinition(31, "Pacific Avenue", "property", 9, 300, 26, 130, 390, 900, 1100, 1275, 200, 150, 0, "Green"),
    SpaceDefinition(32, "North Carolina Avenue", "property", 9, 300, 26, 130, 390, 900, 1100, 1275, 200, 150, 0, "Green"),
    SpaceDefinition(33, "Community Chest", "community_chest"),
    SpaceDefinition(34, "Pennsylvania Avenue", "property", 9, 320, 28, 150, 450, 1000, 1200, 1400, 200, 160, 0, "Green"),
    SpaceDefinition(35, "Short Line", "railroad", 1, 200, 0, 0, 0, 0, 0, 0, 0, 100),
    SpaceDefinition(36, "Chance", "chance"),
    SpaceDefinition(37, "Park Place", "property", 10, 350, 35, 175, 500, 1100, 1300, 1500, 200, 175, 0, "Dark Blue"),
    SpaceDefinition(38, "Luxury Tax", "tax", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100),
    SpaceDefinition(39, "Boardwalk", "property", 10, 400, 50, 200, 600, 1400, 1700, 2000, 200, 200, 0, "Dark Blue"),
]

GROUP_MEMBERS = {}
for space in BOARD:
    if space.group > 0:
        if space.group not in GROUP_MEMBERS:
            GROUP_MEMBERS[space.group] = []
        GROUP_MEMBERS[space.group].append(space.index)


@dataclass(frozen=True)
class CardDefinition:
    index: int
    deck: str
    text: str
    effect_type: str
    amount: int = 0
    destination: int | None = None
    per_house: int = 0
    per_hotel: int = 0

COMMUNITY_CHEST_CARDS: list[CardDefinition] = [
    CardDefinition(0, "community_chest", "Get out of Jail Free", "jail_free"),
    CardDefinition(1, "community_chest", "Beauty contest, collect $10", "collect", amount=10),
    CardDefinition(2, "community_chest", "Sale of stock, get $50", "collect", amount=50),
    CardDefinition(3, "community_chest", "Life insurance matures, $100", "collect", amount=100),
    CardDefinition(4, "community_chest", "Income tax refund, $20", "collect", amount=20),
    CardDefinition(5, "community_chest", "Holiday fund matures, $100", "collect", amount=100),
    CardDefinition(6, "community_chest", "Inherit $100", "collect", amount=100),
    CardDefinition(7, "community_chest", "Consultancy fee, $25", "collect", amount=25),
    CardDefinition(8, "community_chest", "Hospital fees $100", "pay", amount=100),
    CardDefinition(9, "community_chest", "Bank error, collect $200", "collect", amount=200),
    CardDefinition(10, "community_chest", "School fees $50", "pay", amount=50),
    CardDefinition(11, "community_chest", "Doctor's fee $50", "pay", amount=50),
    CardDefinition(12, "community_chest", "Birthday, collect $10 from each player", "collect_from_each", amount=10),
    CardDefinition(13, "community_chest", "Advance to GO", "move", destination=0),
    CardDefinition(14, "community_chest", "Street repairs: $40/house, $115/hotel", "repairs", per_house=40, per_hotel=115),
    CardDefinition(15, "community_chest", "Go to Jail", "jail"),
]

CHANCE_CARDS: list[CardDefinition] = [
    CardDefinition(0, "chance", "Get out of Jail Free", "jail_free"),
    CardDefinition(1, "chance", "General repairs: $25/house, $100/hotel", "repairs", per_house=25, per_hotel=100),
    CardDefinition(2, "chance", "Speeding fine $15", "pay", amount=15),
    CardDefinition(3, "chance", "Chairman of the board, pay each player $50", "pay_each", amount=50),
    CardDefinition(4, "chance", "Go back 3 spaces", "go_back", amount=3),
    CardDefinition(5, "chance", "Advance to nearest utility", "move_nearest", destination=2), # group 2 is utility
    CardDefinition(6, "chance", "Bank dividend $50", "collect", amount=50),
    CardDefinition(7, "chance", "Advance to nearest railroad", "move_nearest", destination=1), # group 1 is railroad
    CardDefinition(8, "chance", "Poor tax $15", "pay", amount=15),
    CardDefinition(9, "chance", "Trip to Reading Railroad", "move", destination=5),
    CardDefinition(10, "chance", "Advance to Boardwalk", "move", destination=39),
    CardDefinition(11, "chance", "Advance to Illinois Ave", "move", destination=24),
    CardDefinition(12, "chance", "Building loan matures $150", "collect", amount=150),
    CardDefinition(13, "chance", "Advance to nearest railroad", "move_nearest", destination=1), # duplicate
    CardDefinition(14, "chance", "Advance to St. Charles Place", "move", destination=11),
    CardDefinition(15, "chance", "Go to Jail", "jail"),
]
