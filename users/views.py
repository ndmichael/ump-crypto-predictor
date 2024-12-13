from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


@login_required
def user_dashboard(request):
    context ={
        "title": "user dashboard"
    }
    return render(request, "users/user_dashboard.html", context)

@login_required
def redirect_dashboard(request):
    return redirect('user_dashboard')
    # if request.user.is_staff:
    #     return redirect('admindashboard')
    