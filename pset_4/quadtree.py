class QuadTreeNode:
    """Constructs a quad tree node with the given ranges and points."""
    def __init__(self, x_range, y_range, points=None):
        self.x_range = x_range  # (x_min, x_max)
        self.y_range = y_range  # (y_min, y_max)
        self.points = points if points else []  # Points in this node
        self.children = []  # Child nodes

    """Returns true if this node is a leaf node, false otherwise."""
    def is_leaf(self):
        return len(self.children) == 0

    """Returns true if this node contains the given point, false otherwise."""
    def contains(self, x, y):
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]

    """Inserts the given point into this node, if possible. Returns true if
    the point was inserted, false otherwise."""
    def insert(self, point):
        if not self.contains_point(point[0], point[1]):
            return False  # Point does not belong to this node

        if self.is_leaf() and len(self.points) < 4:  # Maximum of 4 points per leaf node
            self.points.append(point)
            return True

        if self.is_leaf():
            self.split()

        return any(child.insert(point) for child in self.children)

    """Splits this node into 4 child nodes, and inserts all points into the
    child nodes."""
    def split(self):
        x_mid = (self.x_range[0] + self.x_range[1]) / 2
        y_mid = (self.y_range[0] + self.y_range[1]) / 2

        self.children = [
            QuadTreeNode((self.x_range[0], x_mid), (self.y_range[0], y_mid)),  # Bottom left
            QuadTreeNode((self.x_range[0], x_mid), (y_mid, self.y_range[1])),  # Bottom right
            QuadTreeNode((x_mid, self.x_range[1]), (self.y_range[0], y_mid)),  # Top left
            QuadTreeNode((x_mid, self.x_range[1]), (y_mid, self.y_range[1]))  # Top right
        ]

        for point in self.points:
            for child in self.children:
                if child.insert(point):
                    break

        self.points = []  # Delete points from this node since they are now in child nodes
