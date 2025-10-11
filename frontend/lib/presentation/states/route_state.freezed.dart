// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'route_state.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$RouteState {

 Location? get start; Location? get end; RouteOption get selectedOption; List<RouteInfo> get routes;
/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$RouteStateCopyWith<RouteState> get copyWith => _$RouteStateCopyWithImpl<RouteState>(this as RouteState, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is RouteState&&(identical(other.start, start) || other.start == start)&&(identical(other.end, end) || other.end == end)&&(identical(other.selectedOption, selectedOption) || other.selectedOption == selectedOption)&&const DeepCollectionEquality().equals(other.routes, routes));
}


@override
int get hashCode => Object.hash(runtimeType,start,end,selectedOption,const DeepCollectionEquality().hash(routes));

@override
String toString() {
  return 'RouteState(start: $start, end: $end, selectedOption: $selectedOption, routes: $routes)';
}


}

/// @nodoc
abstract mixin class $RouteStateCopyWith<$Res>  {
  factory $RouteStateCopyWith(RouteState value, $Res Function(RouteState) _then) = _$RouteStateCopyWithImpl;
@useResult
$Res call({
 Location? start, Location? end, RouteOption selectedOption, List<RouteInfo> routes
});


$LocationCopyWith<$Res>? get start;$LocationCopyWith<$Res>? get end;

}
/// @nodoc
class _$RouteStateCopyWithImpl<$Res>
    implements $RouteStateCopyWith<$Res> {
  _$RouteStateCopyWithImpl(this._self, this._then);

  final RouteState _self;
  final $Res Function(RouteState) _then;

/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? start = freezed,Object? end = freezed,Object? selectedOption = null,Object? routes = null,}) {
  return _then(_self.copyWith(
start: freezed == start ? _self.start : start // ignore: cast_nullable_to_non_nullable
as Location?,end: freezed == end ? _self.end : end // ignore: cast_nullable_to_non_nullable
as Location?,selectedOption: null == selectedOption ? _self.selectedOption : selectedOption // ignore: cast_nullable_to_non_nullable
as RouteOption,routes: null == routes ? _self.routes : routes // ignore: cast_nullable_to_non_nullable
as List<RouteInfo>,
  ));
}
/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get start {
    if (_self.start == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.start!, (value) {
    return _then(_self.copyWith(start: value));
  });
}/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get end {
    if (_self.end == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.end!, (value) {
    return _then(_self.copyWith(end: value));
  });
}
}


/// Adds pattern-matching-related methods to [RouteState].
extension RouteStatePatterns on RouteState {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _SelectedRoute value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _SelectedRoute() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _SelectedRoute value)  $default,){
final _that = this;
switch (_that) {
case _SelectedRoute():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _SelectedRoute value)?  $default,){
final _that = this;
switch (_that) {
case _SelectedRoute() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( Location? start,  Location? end,  RouteOption selectedOption,  List<RouteInfo> routes)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _SelectedRoute() when $default != null:
return $default(_that.start,_that.end,_that.selectedOption,_that.routes);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( Location? start,  Location? end,  RouteOption selectedOption,  List<RouteInfo> routes)  $default,) {final _that = this;
switch (_that) {
case _SelectedRoute():
return $default(_that.start,_that.end,_that.selectedOption,_that.routes);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( Location? start,  Location? end,  RouteOption selectedOption,  List<RouteInfo> routes)?  $default,) {final _that = this;
switch (_that) {
case _SelectedRoute() when $default != null:
return $default(_that.start,_that.end,_that.selectedOption,_that.routes);case _:
  return null;

}
}

}

/// @nodoc


class _SelectedRoute implements RouteState {
  const _SelectedRoute({this.start, this.end, required this.selectedOption, required final  List<RouteInfo> routes}): _routes = routes;
  

@override final  Location? start;
@override final  Location? end;
@override final  RouteOption selectedOption;
 final  List<RouteInfo> _routes;
@override List<RouteInfo> get routes {
  if (_routes is EqualUnmodifiableListView) return _routes;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_routes);
}


/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$SelectedRouteCopyWith<_SelectedRoute> get copyWith => __$SelectedRouteCopyWithImpl<_SelectedRoute>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _SelectedRoute&&(identical(other.start, start) || other.start == start)&&(identical(other.end, end) || other.end == end)&&(identical(other.selectedOption, selectedOption) || other.selectedOption == selectedOption)&&const DeepCollectionEquality().equals(other._routes, _routes));
}


@override
int get hashCode => Object.hash(runtimeType,start,end,selectedOption,const DeepCollectionEquality().hash(_routes));

@override
String toString() {
  return 'RouteState(start: $start, end: $end, selectedOption: $selectedOption, routes: $routes)';
}


}

/// @nodoc
abstract mixin class _$SelectedRouteCopyWith<$Res> implements $RouteStateCopyWith<$Res> {
  factory _$SelectedRouteCopyWith(_SelectedRoute value, $Res Function(_SelectedRoute) _then) = __$SelectedRouteCopyWithImpl;
@override @useResult
$Res call({
 Location? start, Location? end, RouteOption selectedOption, List<RouteInfo> routes
});


@override $LocationCopyWith<$Res>? get start;@override $LocationCopyWith<$Res>? get end;

}
/// @nodoc
class __$SelectedRouteCopyWithImpl<$Res>
    implements _$SelectedRouteCopyWith<$Res> {
  __$SelectedRouteCopyWithImpl(this._self, this._then);

  final _SelectedRoute _self;
  final $Res Function(_SelectedRoute) _then;

/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? start = freezed,Object? end = freezed,Object? selectedOption = null,Object? routes = null,}) {
  return _then(_SelectedRoute(
start: freezed == start ? _self.start : start // ignore: cast_nullable_to_non_nullable
as Location?,end: freezed == end ? _self.end : end // ignore: cast_nullable_to_non_nullable
as Location?,selectedOption: null == selectedOption ? _self.selectedOption : selectedOption // ignore: cast_nullable_to_non_nullable
as RouteOption,routes: null == routes ? _self._routes : routes // ignore: cast_nullable_to_non_nullable
as List<RouteInfo>,
  ));
}

/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get start {
    if (_self.start == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.start!, (value) {
    return _then(_self.copyWith(start: value));
  });
}/// Create a copy of RouteState
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get end {
    if (_self.end == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.end!, (value) {
    return _then(_self.copyWith(end: value));
  });
}
}

// dart format on
