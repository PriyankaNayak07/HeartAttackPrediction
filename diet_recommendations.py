def get_diet_recommendations(prediction, age, sex):
    """
    Generate personalized diet recommendations based on prediction result and user demographics
    
    Args:
        prediction: Heart disease prediction (0 or 1)
        age: User's age
        sex: User's biological sex (Male/Female)
        
    Returns:
        Dictionary containing diet recommendations
    """
    # Base recommendations for everyone
    base_foods_to_include = [
        "Fresh fruits and vegetables (aim for 5+ servings daily)",
        "Whole grains (brown rice, oats, quinoa, whole wheat)",
        "Lean proteins (fish, poultry, legumes, tofu)",
        "Nuts and seeds (almonds, walnuts, flaxseeds, chia seeds)",
        "Healthy oils (olive oil, avocado oil)"
    ]
    
    base_foods_to_limit = [
        "Processed and red meats",
        "Foods high in added sugars",
        "Highly processed foods and snacks",
        "Foods high in saturated and trans fats",
        "High-sodium foods and added salt"
    ]
    
    # General guidelines for everyone
    general_guidelines = """
    A heart-healthy diet focuses on foods that can help lower cholesterol, manage blood pressure, 
    maintain healthy weight, and reduce inflammation. It's rich in fruits, vegetables, whole grains, 
    and lean proteins while limiting sodium, unhealthy fats, and added sugars.
    """
    
    # Sample meal plan base
    sample_meal_plan = """
    **Breakfast**: Oatmeal with berries and nuts, or whole grain toast with avocado
    
    **Lunch**: Salad with mixed greens, vegetables, grilled chicken or tofu, and olive oil dressing
    
    **Dinner**: Baked fish or legumes with steamed vegetables and quinoa or brown rice
    
    **Snacks**: Fresh fruit, Greek yogurt, or a small handful of unsalted nuts
    """
    
    # Modify recommendations based on prediction and demographics
    if prediction == 1:  # Predicted heart disease
        general_guidelines += """
        
        With your heart health risk factors, it's particularly important to follow a heart-protective 
        diet that helps manage cholesterol and blood pressure.
        """
        
        additional_foods = [
            "Fatty fish rich in omega-3s (salmon, mackerel, sardines) - 2-3 servings weekly",
            "Berries rich in antioxidants (blueberries, strawberries)",
            "Heart-healthy fats (avocados, olive oil)",
            "Foods high in soluble fiber (oats, beans, apples)"
        ]
        
        additional_limits = [
            "Limit sodium to less than 1,500 mg daily",
            "Avoid trans fats completely",
            "Limit alcohol consumption significantly",
            "Reduce caffeine intake"
        ]
        
        # Add age-specific modifications
        if age > 60:
            additional_foods.append("Calcium-rich foods for bone health (low-fat dairy, fortified plant milks)")
            additional_foods.append("Vitamin D sources (eggs, fortified foods)")
        
        # Add sex-specific modifications
        if sex == "Male":
            additional_limits.append("Limit red meat to once weekly or less")
        else:  # Female
            additional_foods.append("Iron-rich foods (spinach, lentils, fortified cereals)")
        
        foods_to_include = base_foods_to_include + additional_foods
        foods_to_limit = base_foods_to_limit + additional_limits
        
    else:  # No predicted heart disease
        general_guidelines += """
        
        Although your heart health risk appears lower, maintaining a heart-healthy diet is still 
        important for prevention and overall health.
        """
        
        # Add age-specific modifications
        if age > 40:
            base_foods_to_include.append("Foods rich in antioxidants (colorful fruits and vegetables)")
            
        foods_to_include = base_foods_to_include
        foods_to_limit = base_foods_to_limit
    
    return {
        "general_guidelines": general_guidelines,
        "foods_to_include": foods_to_include,
        "foods_to_limit": foods_to_limit,
        "sample_meal_plan": sample_meal_plan
    }
